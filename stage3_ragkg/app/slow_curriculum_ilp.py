from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from ortools.sat.python import cp_model

from .kg.loader import CourseNode
from .logging_utils import get_logger, log_call

# Module-level logger
logger = get_logger(__name__)

# Default configuration for IU thesis and electives
DEFAULT_THESIS_COURSE_IDS: List[str] = ["IT058IU"]
DEFAULT_MIN_ELECTIVE_CREDITS: int = 15

# IU-specific thesis replacement track
IU_THESIS_COURSE_ID: str = "IT058IU"
IU_THESIS_REPLACEMENT_COURSE_ID: str = "IT168IU"
IU_REPLACEMENT_MIN_EXTRA_ELECTIVE_COURSES: int = 2
IU_REPLACEMENT_MIN_EXTRA_ELECTIVE_CREDITS: int = 7

@dataclass
class UserCurriculumConstraints:
    """
    User-level constraints for curriculum planning.

    - total_years * semesters_per_year defines the total number of terms,
      unless terms_remaining is given explicitly.
    - max_credits_per_semester is a hard upper bound.
    - preferred_courses are treated as "should be taken" (like compulsory).
    - avoid_courses are never scheduled.
    - completed_courses are already finished by the student and will not be scheduled again.

    - current_semester_index represents the IU program semester number for the
      *next* registration:
        * 1  -> the student is just starting the program
        * 3  -> the student has already completed semesters 1 and 2, and the
                next semester in the curriculum is semester 3
    """
    # IU curriculum structure
    total_years: int
    semesters_per_year: int
    max_credits_per_semester: int = 24
    min_credits_per_semester: int = 14

    # Course-level preferences
    preferred_courses: List[str] = field(default_factory=list)
    avoid_courses: List[str] = field(default_factory=list)

    # Optional: directly specify the number of semesters to plan for (from "now")
    terms_remaining: Optional[int] = None

    # IU program semester index for the *next* semester the student will register for (1..8)
    current_semester_index: int = 1

    # Courses that the student has already completed; they will not be scheduled again
    completed_courses: List[str] = field(default_factory=list)
    
    # IU-specific thesis handling
    # If empty, DEFAULT_THESIS_COURSE_IDS will be used.
    thesis_course_ids: List[str] = field(default_factory=list)

    # Minimum total elective credits required by IU
    # If set to <= 0, the elective-credit constraint will be skipped.
    min_total_elective_credits: int = DEFAULT_MIN_ELECTIVE_CREDITS

    # If True: replace Thesis (IT058IU) with IT168IU + an additional elective bundle.
    # Default False keeps legacy behavior: Thesis is required.
    use_thesis_replacement: bool = False

@dataclass
class PlannedSemester:
    index: int
    courses: List["PlannedCourse"]
    total_credits: int


@dataclass
class PlannedCourse:
    """A scheduled course with lightweight metadata for explanation/UI."""

    id: str
    name: str
    credits: Optional[int] = None

@dataclass
class CurriculumPlan:
    status: Literal["OK", "INFEASIBLE", "ERROR"]
    semesters: List[PlannedSemester]
    message: Optional[str] = None

class CurriculumPlanningError(Exception):
    """Raised when the planning pipeline fails for non-ILP reasons."""

@dataclass
class CurriculumGraph:
    """
    Lightweight abstraction over the curriculum KG for planning.

    courses:
        id -> CourseNode
    prereq_in:
        course_id -> list of prerequisite course_ids
    """
    courses: Dict[str, CourseNode]
    prereq_in: Dict[str, List[str]]


def _is_semester_offered(course: CourseNode, term_index: int) -> bool:
    """
    Check whether a course is offered in a given (1-based) term index.

    CourseNode.semester is either "all" or a list of positive ints.
    For now we assume term_index corresponds directly to this integer.
    """
    sem = course.semester
    if sem == "all":
        return True
    # pydantic model uses List[int] for non-"all"
    return term_index in sem  # type: ignore[arg-type]


def _course_credits(course: CourseNode) -> int:
    return int(course.credits.total)


@log_call(level=10, include_result=False)
def build_curriculum_graph(
    node_map: Dict[str, CourseNode],
    prereq_in: Dict[str, List[str]],
    avoid_courses: Optional[List[str]] = None,
) -> CurriculumGraph:
    """
    Convert raw KG structures into CurriculumGraph, optionally filtering out avoid_courses.
    """
    avoid_set = {c.upper() for c in (avoid_courses or [])}
    logger.debug(
        "build_curriculum_graph: starting with %d nodes, avoid_courses=%s",
        len(node_map),
        sorted(avoid_set),
    )
    filtered_courses: Dict[str, CourseNode] = {
        cid: node for cid, node in node_map.items() if cid.upper() not in avoid_set
    }
    # Filter prereq_in to only reference kept courses
    filtered_prereq_in: Dict[str, List[str]] = {}
    edge_count = 0
    for cid, preds in prereq_in.items():
        if cid.upper() not in avoid_set and cid in filtered_courses:
            kept_preds = [p for p in preds if p in filtered_courses]
            filtered_prereq_in[cid] = kept_preds
            edge_count += len(kept_preds)
    logger.info(
        "build_curriculum_graph: resulting courses=%d edges=%d",
        len(filtered_courses),
        edge_count,
    )
    return CurriculumGraph(courses=filtered_courses, prereq_in=filtered_prereq_in)


def _add_course_selection_constraints(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    user_constraints: UserCurriculumConstraints,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> None:
    """
    Each course can be taken at most once.
    Compulsory + preferred courses must be taken exactly once.
    """
    preferred = {c.upper() for c in user_constraints.preferred_courses}
    configured_thesis_ids = {
        cid.upper()
        for cid in (user_constraints.thesis_course_ids or DEFAULT_THESIS_COURSE_IDS)
    }
    # Thesis & its replacement are handled by dedicated constraints.
    special_exempt_ids = set(configured_thesis_ids) | {IU_THESIS_REPLACEMENT_COURSE_ID.upper()}
    logger.debug(
        "_add_course_selection_constraints: preferred=%s total_courses=%d",
        sorted(preferred),
        len(graph.courses),
    )
    for cid, course in graph.courses.items():
        vars_for_course = [x[(cid, s)] for s in all_semesters]
        is_compulsory = course.compulsory == "Compulsory"
        is_preferred = cid.upper() in preferred
        if cid.upper() in special_exempt_ids:
            # Do not force these via compulsory/preferred rules; handled elsewhere.
            model.Add(sum(vars_for_course) <= 1)
        elif is_compulsory or is_preferred:
            model.Add(sum(vars_for_course) == 1)
        else:
            model.Add(sum(vars_for_course) <= 1)


def _add_availability_constraints(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> None:
    """
    Enforce course availability:
      if a course is not offered in a term, its x[c, term] must be 0.
    """
    logger.debug(
        "_add_availability_constraints: semesters=%s courses=%d",
        all_semesters,
        len(graph.courses),
    )
    for cid, course in graph.courses.items():
        for s in all_semesters:
            if not _is_semester_offered(course, s):
                model.Add(x[(cid, s)] == 0)


def _add_prerequisite_constraints(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> None:
    """
    Prerequisite ordering:

    For each edge a -> b, we enforce:
      - if b is taken in semester s, then a must be taken in some semester < s.
    """
    logger.debug(
        "_add_prerequisite_constraints: courses_with_prereq=%d",
        sum(1 for preds in graph.prereq_in.values() if preds),
    )
    for b, preds in graph.prereq_in.items():
        if not preds:
            continue
        for s in all_semesters:
            b_var = x[(b, s)]
            if s == 1:
                # cannot take course with prerequisites in first semester
                model.Add(b_var == 0)
            else:
                for a in preds:
                    sum_prev = sum(x[(a, t)] for t in all_semesters if t < s)
                    # If b is taken in s, some a must be taken before s.
                    model.Add(sum_prev >= b_var)


def _add_credit_constraints(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    user_constraints: UserCurriculumConstraints,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> None:
    """
    Semester credit limits:

      For each term s:
        sum_{c} credits_c * x[c, s] <= max_credits_per_semester

    If min_credits_per_semester is set:
        sum_{c} credits_c * x[c, s] >= min_credits_per_semester
    for all semesters EXCEPT the final semester in the planning horizon.
    """
    if not all_semesters:
        logger.warning("_add_credit_constraints: no semesters to constrain.")
        return

    max_credits = int(user_constraints.max_credits_per_semester)
    min_credits = (
        int(user_constraints.min_credits_per_semester)
        if user_constraints.min_credits_per_semester is not None
        else None
    )

    last_planned_semester = all_semesters[-1]
    logger.debug(
        "_add_credit_constraints: semesters=%s max=%d min=%s last_sem=%d",
        all_semesters,
        max_credits,
        min_credits,
        last_planned_semester,
    )

    for s in all_semesters:
        term_credits = sum(
            _course_credits(graph.courses[cid]) * x[(cid, s)]
            for cid in graph.courses
        )

        # Hard upper bound for every semester
        model.Add(term_credits <= max_credits)

        # Minimum load per semester (IU rule) – but skip the final semester
        if (
            min_credits is not None
            and min_credits > 0
            and s != last_planned_semester
        ):
            model.Add(term_credits >= min_credits)


def _add_elective_credit_constraints(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    user_constraints: UserCurriculumConstraints,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> None:
    """
    Global constraint: total elective credits >= min_total_elective_credits.

    elective_credits = sum_{c elective} sum_{s} credits_c * x[c, s]
    """
    min_elective = int(user_constraints.min_total_elective_credits)
    if getattr(user_constraints, "use_thesis_replacement", False):
        # In replacement mode we enforce a stricter, disjoint elective-bundle rule
        # (baseline electives + replacement electives). Skip the simple global-credit rule.
        logger.info(
            "_add_elective_credit_constraints: use_thesis_replacement=True; skipping global elective_credits >= %d",
            min_elective,
        )
        return
    if min_elective <= 0:
        logger.info(
            "_add_elective_credit_constraints: min_total_elective_credits=%d, skipping.",
            min_elective,
        )
        return

    elective_course_ids = [
        cid for cid, course in graph.courses.items()
        if getattr(course, "compulsory", None) == "Elective"
    ]
    if not elective_course_ids:
        logger.warning(
            "_add_elective_credit_constraints: no elective courses found, "
            "cannot enforce elective credits >= %d",
            min_elective,
        )
        return

    elective_credits_expr = sum(
        _course_credits(graph.courses[cid]) * x[(cid, s)]
        for cid in elective_course_ids
        for s in all_semesters
    )

    logger.info(
        "_add_elective_credit_constraints: enforcing elective_credits >= %d over %d electives.",
        min_elective,
        len(elective_course_ids),
    )
    model.Add(elective_credits_expr >= min_elective)


def _add_thesis_or_replacement_constraints(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    user_constraints: UserCurriculumConstraints,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> None:
    """IU-specific track selection between Thesis and the replacement bundle.

    Modes:
      - use_thesis_replacement=False (default):
          * require Thesis (IT058IU)
          * forbid IT168IU
          * elective rules are handled by _add_elective_credit_constraints (>=15)

      - use_thesis_replacement=True:
          * forbid Thesis (IT058IU)
          * require IT168IU
          * additionally require:
              - baseline electives: >= min_total_elective_credits (default 15)
              - replacement electives: >=2 distinct elective courses AND >=7 elective credits
              - baseline electives and replacement electives must be disjoint
    """

    use_replacement = bool(getattr(user_constraints, "use_thesis_replacement", False))

    thesis_id = None
    for cid in graph.courses:
        if cid.upper() == IU_THESIS_COURSE_ID.upper():
            thesis_id = cid
            break

    replacement_id = None
    for cid in graph.courses:
        if cid.upper() == IU_THESIS_REPLACEMENT_COURSE_ID.upper():
            replacement_id = cid
            break

    if use_replacement:
        if replacement_id is None:
            raise CurriculumPlanningError(
                f"Replacement mode requires {IU_THESIS_REPLACEMENT_COURSE_ID}, but it was not found in the KG."
            )

        # Require IT168IU
        model.Add(sum(x[(replacement_id, s)] for s in all_semesters) == 1)

        # Forbid thesis if present in KG (preferred)
        if thesis_id is not None:
            model.Add(sum(x[(thesis_id, s)] for s in all_semesters) == 0)

        logger.info(
            "_add_thesis_or_replacement_constraints: replacement mode ON -> require %s, forbid %s",
            replacement_id,
            thesis_id or IU_THESIS_COURSE_ID,
        )

        # --- Elective bundle requirements (disjoint buckets) ---
        elective_course_ids = [
            cid
            for cid, course in graph.courses.items()
            if getattr(course, "compulsory", None) == "Elective"
        ]
        if not elective_course_ids:
            raise CurriculumPlanningError(
                "Replacement mode requires electives, but no elective courses were detected in the KG."
            )

        baseline_min = int(user_constraints.min_total_elective_credits or DEFAULT_MIN_ELECTIVE_CREDITS)
        baseline_min = max(baseline_min, DEFAULT_MIN_ELECTIVE_CREDITS)

        base_bucket: Dict[str, cp_model.IntVar] = {}
        repl_bucket: Dict[str, cp_model.IntVar] = {}

        for cid in elective_course_ids:
            base_bucket[cid] = model.NewBoolVar(f"elective_base_{cid}")
            repl_bucket[cid] = model.NewBoolVar(f"elective_repl_{cid}")

            # A course can only belong to one bucket.
            model.Add(base_bucket[cid] + repl_bucket[cid] <= 1)

            # Buckets can only include courses that are actually taken.
            taken_expr = sum(x[(cid, s)] for s in all_semesters)  # 0..1
            model.Add(base_bucket[cid] <= taken_expr)
            model.Add(repl_bucket[cid] <= taken_expr)

        base_credits_expr = sum(_course_credits(graph.courses[cid]) * base_bucket[cid] for cid in elective_course_ids)
        repl_credits_expr = sum(_course_credits(graph.courses[cid]) * repl_bucket[cid] for cid in elective_course_ids)
        repl_count_expr = sum(repl_bucket[cid] for cid in elective_course_ids)

        model.Add(base_credits_expr >= baseline_min)
        model.Add(repl_credits_expr >= IU_REPLACEMENT_MIN_EXTRA_ELECTIVE_CREDITS)
        model.Add(repl_count_expr >= IU_REPLACEMENT_MIN_EXTRA_ELECTIVE_COURSES)

        logger.info(
            "_add_thesis_or_replacement_constraints: replacement electives -> base_credits>=%d AND repl_count>=%d AND repl_credits>=%d (disjoint)",
            baseline_min,
            IU_REPLACEMENT_MIN_EXTRA_ELECTIVE_COURSES,
            IU_REPLACEMENT_MIN_EXTRA_ELECTIVE_CREDITS,
        )

    else:
        # Legacy/default mode: require Thesis and forbid IT168IU.
        if thesis_id is None:
            raise CurriculumPlanningError(
                f"Thesis mode requires {IU_THESIS_COURSE_ID}, but it was not found in the KG."
            )

        model.Add(sum(x[(thesis_id, s)] for s in all_semesters) == 1)

        if replacement_id is not None:
            model.Add(sum(x[(replacement_id, s)] for s in all_semesters) == 0)

        logger.info(
            "_add_thesis_or_replacement_constraints: thesis mode ON -> require %s, forbid %s",
            thesis_id,
            replacement_id or IU_THESIS_REPLACEMENT_COURSE_ID,
        )


def _add_thesis_last_semester_constraints(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    user_constraints: UserCurriculumConstraints,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> None:
    """
    if getattr(user_constraints, "use_thesis_replacement", False):
        logger.info(
            "_add_thesis_last_semester_constraints: use_thesis_replacement=True; skipping thesis-last-semester rule."
        )
        return
    Enforce that once the thesis is taken in some semester s, there are no courses
    in any later semester t > s.

    This ensures:
      - Thesis is always in the last active semester.
      - Any courses taken with thesis in that semester are indeed the "final remaining"
        courses of the plan.
    """
    configured_thesis_ids = {
        cid.upper() for cid in (user_constraints.thesis_course_ids or DEFAULT_THESIS_COURSE_IDS)
    }

    thesis_ids_in_graph = [
        cid for cid in graph.courses if cid.upper() in configured_thesis_ids
    ]
    if not thesis_ids_in_graph:
        logger.warning(
            "_add_thesis_last_semester_constraints: no thesis course found in graph for ids=%s",
            sorted(configured_thesis_ids),
        )
        return

    if len(thesis_ids_in_graph) > 1:
        logger.warning(
            "_add_thesis_last_semester_constraints: multiple thesis candidates found: %s; using the first.",
            thesis_ids_in_graph,
        )

    thesis_id = thesis_ids_in_graph[0]
    logger.info(
        "_add_thesis_last_semester_constraints: thesis_id=%s, semesters=%s",
        thesis_id,
        all_semesters,
    )

    for s in all_semesters:
        # If thesis is taken in semester s, forbid any other course in later semesters.
        thesis_in_s = x[(thesis_id, s)]
        for t in all_semesters:
            if t <= s:
                continue
            for cid in graph.courses:
                if cid == thesis_id:
                    continue
                # x[cid, t] <= 1 - x[thesis_id, s]
                model.Add(x[(cid, t)] <= 1 - thesis_in_s)


def _add_objective_minimize_last_semester(
    model: cp_model.CpModel,
    graph: CurriculumGraph,
    x: Dict[Tuple[str, int], cp_model.IntVar],
    all_semesters: List[int],
) -> cp_model.IntVar:
    """
    Objective: minimize the last semester index that has any course,
    with a small extra penalty if *other* courses share a semester with
    the thesis course.

    Hard preferences:
      - Finish as early as possible (minimize last_semester).

    Soft preference:
      - If two plans finish in the same last_semester, prefer the one
        where the thesis is more "alone" in its semester (fewer other
        courses taken in the same term).
    """
    if not all_semesters:
        raise CurriculumPlanningError("No semesters defined in objective.")

    logger.debug(
        "_add_objective_minimize_last_semester: semesters=%s courses=%d",
        all_semesters,
        len(graph.courses),
    )

    # --- Part 1: original "finish early" objective setup ---

    num_courses = len(graph.courses)
    y: Dict[int, cp_model.IntVar] = {}

    for s in all_semesters:
        y_s = model.NewBoolVar(f"any_course_sem_{s}")
        y[s] = y_s

        # sum_c x[c,s] >= y_s
        sum_x = sum(x[(cid, s)] for cid in graph.courses)
        model.Add(sum_x >= y_s)
        # sum_c x[c,s] <= num_courses * y_s
        model.Add(sum_x <= num_courses * y_s)

    last_semester = model.NewIntVar(0, max(all_semesters), "last_semester")
    for s in all_semesters:
        model.Add(last_semester >= s * y[s])

    # --- Part 2: soft penalty if other courses share the thesis semester ---

    # Try to locate the thesis course in the graph
    configured_thesis_ids = {cid.upper() for cid in DEFAULT_THESIS_COURSE_IDS}
    thesis_ids_in_graph = [
        cid for cid in graph.courses
        if cid.upper() in configured_thesis_ids
    ]

    if not thesis_ids_in_graph:
        # No thesis course present -> fall back to pure "finish early" objective
        logger.info(
            "_add_objective_minimize_last_semester: no thesis course found "
            "for ids=%s; using pure last_semester objective.",
            sorted(configured_thesis_ids),
        )
        model.Minimize(last_semester)
        return last_semester

    # Use the first matching thesis id (IU case: just IT058IU)
    thesis_id = thesis_ids_in_graph[0]
    logger.info(
        "_add_objective_minimize_last_semester: using thesis_id=%s for soft penalty.",
        thesis_id,
    )

    # co_with_thesis[c,s] = 1 if course c (!= thesis) is taken in the same
    # semester as the thesis.
    co_with_thesis: List[cp_model.IntVar] = []

    for s in all_semesters:
        thesis_in_s = x[(thesis_id, s)]
        for cid in graph.courses:
            if cid == thesis_id:
                continue

            both = model.NewBoolVar(f"course_{cid}_with_thesis_sem_{s}")

            # both <= x[cid,s]
            model.Add(both <= x[(cid, s)])
            # both <= x[thesis_id,s]
            model.Add(both <= thesis_in_s)
            # both >= x[cid,s] + x[thesis_id,s] - 1
            model.Add(both >= x[(cid, s)] + thesis_in_s - 1)

            co_with_thesis.append(both)

    if not co_with_thesis:
        # Graph only has thesis or something degenerate -> just minimize last_semester
        logger.info(
            "_add_objective_minimize_last_semester: no non-thesis courses; "
            "using pure last_semester objective."
        )
        model.Minimize(last_semester)
        return last_semester

    total_co_with_thesis = sum(co_with_thesis)

    # Big weight for "finish early", tiny weight for "thesis alone"
    BIG_WEIGHT = 1000
    PENALTY_WEIGHT = 1

    objective_expr = BIG_WEIGHT * last_semester + PENALTY_WEIGHT * total_co_with_thesis
    model.Minimize(objective_expr)

    return last_semester


@log_call(level=20, include_result=False)
def plan_curriculum_slow(
    node_map: Dict[str, CourseNode],
    prereq_in: Dict[str, List[str]],
    user_constraints: UserCurriculumConstraints,
) -> CurriculumPlan:
    """
    Build and solve an ILP/CP model for curriculum planning.

    Args:
        node_map: course_id -> CourseNode from app.kg.loader.load_graph
        prereq_in: course_id -> list of prerequisite ids
        user_constraints: UserCurriculumConstraints

    Returns:
        CurriculumPlan with status:
          - "OK" if a plan was found,
          - "INFEASIBLE" if constraints could not be satisfied,
          - "ERROR" for other failures.
    """
    logger.info(
        "plan_curriculum_slow: starting with courses=%d constraints={min=%s, max=%s, terms_remaining=%s, current_sem_index=%s, completed=%d, avoid=%d, use_thesis_replacement=%s}",
        len(node_map),
        getattr(user_constraints, "min_credits_per_semester", None),
        getattr(user_constraints, "max_credits_per_semester", None),
        getattr(user_constraints, "terms_remaining", None),
        getattr(user_constraints, "current_semester_index", None),
        len(user_constraints.completed_courses or []),
        len(user_constraints.avoid_courses or []),
        bool(getattr(user_constraints, "use_thesis_replacement", False)),
    )
    try:
        # Determine the IU semester index for the next term the student will register for.
        # We only support 8 IU program semesters (1..8).
        start_sem = int(user_constraints.current_semester_index)
        if start_sem < 1 or start_sem > 8:
            raise CurriculumPlanningError("current_semester_index must be between 1 and 8.")

        # Determine how many future semesters we want to plan.
        if user_constraints.terms_remaining is not None:
            if user_constraints.terms_remaining <= 0:
                raise CurriculumPlanningError("terms_remaining must be positive when provided.")
            planned_semesters = int(user_constraints.terms_remaining)
        else:
            # Fall back to total_years * semesters_per_year if terms_remaining is not given.
            if user_constraints.total_years <= 0:
                raise CurriculumPlanningError("total_years must be positive.")
            if user_constraints.semesters_per_year <= 0:
                raise CurriculumPlanningError("semesters_per_year must be positive.")
            planned_semesters = int(
                user_constraints.total_years * user_constraints.semesters_per_year
            )

        if user_constraints.max_credits_per_semester <= 0:
            raise CurriculumPlanningError("max_credits_per_semester must be positive.")

        # Convert "how many semesters" into a concrete list of IU semesters, clamped to [1..8].
        end_sem = min(8, start_sem + planned_semesters - 1)
        if end_sem < start_sem:
            raise CurriculumPlanningError("No valid semesters in planning horizon.")

        all_semesters = list(range(start_sem, end_sem + 1))
        logger.info(
            "plan_curriculum_slow: planning horizon start_sem=%d end_sem=%d terms=%d",
            start_sem,
            end_sem,
            len(all_semesters),
        )

        # Merge "avoid" and "completed" into one skip-list for scheduling
        avoid_all = list({
            *(user_constraints.avoid_courses or []),
            *(user_constraints.completed_courses or []),
        })
        logger.debug(
            "plan_curriculum_slow: avoid_all merged list size=%d values=%s",
            len(avoid_all),
            sorted({c.upper() for c in avoid_all}),
        )

        graph = build_curriculum_graph(
            node_map=node_map,
            prereq_in=prereq_in,
            avoid_courses=avoid_all,
        )

        if not graph.courses:
            logger.info("plan_curriculum_slow: no courses remain after filtering; returning OK with empty plan.")
            return CurriculumPlan(
                status="OK",
                semesters=[],
                message="No courses to schedule (all courses filtered out by constraints).",
            )

        # --- Build model ---
        model = cp_model.CpModel()

        # Decision variables x[c, s] ∈ {0,1}
        x: Dict[Tuple[str, int], cp_model.IntVar] = {}
        for cid in graph.courses:
            for s in all_semesters:
                x[(cid, s)] = model.NewBoolVar(f"x_{cid}_{s}")

        logger.debug(
            "plan_curriculum_slow: created %d decision variables (courses=%d, semesters=%d)",
            len(x),
            len(graph.courses),
            len(all_semesters),
        )

        # Constraints
        _add_course_selection_constraints(model, graph, user_constraints, x, all_semesters)
        _add_availability_constraints(model, graph, x, all_semesters)
        _add_prerequisite_constraints(model, graph, x, all_semesters)
        _add_credit_constraints(model, graph, user_constraints, x, all_semesters)
        _add_thesis_or_replacement_constraints(model, graph, user_constraints, x, all_semesters)
        _add_elective_credit_constraints(model, graph, user_constraints, x, all_semesters)
        _add_thesis_last_semester_constraints(model, graph, user_constraints, x, all_semesters)
        logger.info("plan_curriculum_slow: added all constraints to the model.")

        # Objective
        last_semester = _add_objective_minimize_last_semester(
            model,
            graph,
            x,
            all_semesters,
        )

        # --- Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 20.0
        solver.parameters.num_search_workers = 8

        logger.info("plan_curriculum_slow: calling CP-SAT solver (time_limit=20s, workers=8)")
        status = solver.Solve(model)
        logger.info("plan_curriculum_slow: solver finished with status=%s", status)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if status == cp_model.INFEASIBLE:
                logger.warning("plan_curriculum_slow: model infeasible")
                return CurriculumPlan(
                    status="INFEASIBLE",
                    semesters=[],
                    message="Constraints are infeasible. Try relaxing max credits or extending total years.",
                )
            logger.error("plan_curriculum_slow: solver failed with non-feasible status=%s", status)
            return CurriculumPlan(
                status="ERROR",
                semesters=[],
                message=f"Solver failed with status {status}.",
            )

        # --- Extract solution ---
        used_last_sem = int(solver.Value(last_semester))
        logger.info("plan_curriculum_slow: used_last_semester=%d", used_last_sem)
        if used_last_sem == 0:
            # No course scheduled despite feasible status → treat as OK but empty.
            logger.warning("plan_curriculum_slow: feasible status but no courses scheduled (used_last_sem=0)")
            return CurriculumPlan(
                status="OK",
                semesters=[],
                message="No courses scheduled by the solver.",
            )

        semesters: List[PlannedSemester] = []
        for s in all_semesters:
            if s > used_last_sem:
                break
            chosen: List[PlannedCourse] = []
            total_credits = 0
            for cid, course in graph.courses.items():
                if solver.Value(x[(cid, s)]) == 1:
                    # Prefer English name if present, else first available name value.
                    name_map = getattr(course, "name", {}) or {}
                    course_name = (
                        (name_map.get("en") if isinstance(name_map, dict) else None)
                        or (next(iter(name_map.values())) if isinstance(name_map, dict) and name_map else None)
                        or cid
                    )
                    c_credits = _course_credits(course)
                    chosen.append(PlannedCourse(id=cid, name=course_name, credits=c_credits))
                    total_credits += c_credits
            chosen.sort(key=lambda x: x.id)
            semesters.append(
                PlannedSemester(
                    index=s,
                    courses=chosen,
                    total_credits=total_credits,
                )
            )

        msg = f"Planned {len(graph.courses)} course(s) over {used_last_sem} semester(s)."
        logger.info(
            "plan_curriculum_slow: built plan semesters=%d msg=%s",
            len(semesters),
            msg,
        )
        return CurriculumPlan(status="OK", semesters=semesters, message=msg)

    except CurriculumPlanningError as exc:
        logger.exception("plan_curriculum_slow: CurriculumPlanningError")
        return CurriculumPlan(
            status="ERROR",
            semesters=[],
            message=f"Planning error: {exc}",
        )
    except Exception as exc:
        logger.exception("plan_curriculum_slow: unexpected error")
        return CurriculumPlan(
            status="ERROR",
            semesters=[],
            message=f"Unexpected error: {exc}",
        )


if __name__ == "__main__":
    # Minimal smoke-test example (you can adapt this into real unit tests).
    from .kg.loader import CourseNode, Credits

    # Simple fake KG: A -> B, both compulsory, both offered in terms 1..4
    ma = CourseNode(
        id="MA101",
        name={"en": "Math 1"},
        credits=Credits(total=3, theory=3, lab=0),
        area=["Core"],
        language="en",
        semester=[1, 2, 3, 4],
        compulsory="Compulsory",
    )
    cs = CourseNode(
        id="CS101",
        name={"en": "Intro CS"},
        credits=Credits(total=3, theory=3, lab=0),
        area=["Core"],
        language="en",
        semester=[2, 3, 4],
        compulsory="Compulsory",
    )
    node_map = {"MA101": ma, "CS101": cs}
    prereq_in = {"MA101": [], "CS101": ["MA101"]}

    constraints = UserCurriculumConstraints(
        total_years=2,
        semesters_per_year=2,
        max_credits_per_semester=6,
    )
    plan = plan_curriculum_slow(node_map, prereq_in, constraints)
    print(plan)
