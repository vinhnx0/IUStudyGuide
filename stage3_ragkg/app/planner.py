from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.llm import llm_generate_json
from app.logging_utils import get_logger, log_call
from app.slow_curriculum_ilp import (
    plan_curriculum_slow as _ilp_plan_curriculum_slow,
    UserCurriculumConstraints as ILPUserCurriculumConstraints,
)
from app.rag.alias_normalizer import AliasNormalizer

logger = get_logger(__name__)

# =====================================================
# Curriculum plan data model
# =====================================================

@dataclass
class CurriculumPlan:
    semesters: List[Dict[str, Any]]
    summary: Dict[str, Any]
    meta: Dict[str, Any]


# =====================================================
# Prompt loader (planner / decompose)
# =====================================================

def _load_planner_prompt_template() -> str:
    candidate_paths = [
        Path("app/prompts/planner_prompt.txt"),
        Path("prompts/planner_prompt.txt"),
        Path("planner_prompt.txt")
    ]
    for path in candidate_paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError("planner_prompt.txt not found")


# =====================================================
# Decomposition (JSON, decompose)
# =====================================================

@log_call(level=logging.INFO, include_result=True)
def decompose_question(question: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    question_clean = (question or "").strip()
    if not question_clean:
        return {
            "entities": [],
            "sub_questions": [],
            "kg_queries": [],
            "meta": {"backend": "none"},
        }

    if not isinstance(cfg, dict):
        return {
            "entities": [],
            "sub_questions": [question_clean],
            "kg_queries": [],
            "meta": {"backend": "invalid_cfg"},
        }

    backend = str((cfg.get("llm") or {}).get("default_backend", "")).lower()
    if backend != "local":
        # decomposition optional → safe fallback
        return {
            "entities": [],
            "sub_questions": [question_clean],
            "kg_queries": [],
            "meta": {"backend": backend},
        }

    try:
        template = _load_planner_prompt_template()
        prompt = template.format(question=question_clean)
    except Exception as exc:
        logger.warning("decompose_question: prompt load failed: %s", exc)
        return {
            "entities": [],
            "sub_questions": [question_clean],
            "kg_queries": [],
            "meta": {"backend": "error"},
        }

    try:
        data = llm_generate_json(
            prompt=prompt,
            cfg=cfg,
            step="decompose",
            max_tokens=800,
        )
    except Exception as exc:
        logger.warning("decompose_question: LLM JSON failed: %s", exc)
        return {
            "entities": [],
            "sub_questions": [question_clean],
            "kg_queries": [],
            "meta": {"backend": "error"},
        }

    return {
        "entities": list(data.get("entities", [])),
        "sub_questions": list(data.get("sub_questions", [])),
        "kg_queries": list(data.get("kg_queries", [])),
        "meta": {"backend": backend},
    }


# =====================================================
# User constraint parsing (heuristic + JSON refine)
# =====================================================

@dataclass
class UserCurriculumConstraints:
    min_credits_per_semester: Optional[int] = None
    max_credits_per_semester: Optional[int] = None
    terms_remaining: Optional[int] = None
    current_semester_index: Optional[int] = None
    completed_semesters: Optional[List[int]] = None
    current_semester_source: Optional[str] = None
    completed_courses: Optional[List[str]] = None
    preferred_courses: Optional[List[str]] = None
    avoid_courses: Optional[List[str]] = None


@log_call(level=logging.INFO, include_result=False)
def parse_user_constraints(
    question: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> UserCurriculumConstraints:
    text = (question or "").strip()

    # -------------------------
    # 1. Heuristic baseline
    # -------------------------
    constraints = UserCurriculumConstraints(
        completed_semesters=[],
        completed_courses=[],
        preferred_courses=[],
        avoid_courses=[],
    )

    credit_re = re.compile(r"(\d+)\s*(?:tín|tin|credits?)", re.IGNORECASE)
    credit_hits = [int(m.group(1)) for m in credit_re.finditer(text)]
    if credit_hits:
        constraints.min_credits_per_semester = min(credit_hits)
        constraints.max_credits_per_semester = max(credit_hits)

    # Only treat this as "terms remaining" when the query explicitly indicates remaining/next terms.
    # Avoid the common pitfall where mentioning "HK1" (completed) accidentally becomes terms_remaining=1.
    remaining_term_re = re.compile(
        r"(?:còn|con|remaining|next|trong)\s*(\d+)\s*(?:kỳ|ki|hoc\s*ky|học\s*kỳ|semesters?|terms?)",
        re.IGNORECASE,
    )
    term_hits = [int(m.group(1)) for m in remaining_term_re.finditer(text)]
    if term_hits:
        constraints.terms_remaining = max(term_hits)

    # Heuristic: current semester phrasing.
    cur_re = re.compile(r"(?:đang\s*học|current)\s*(?:kỳ|kì|hoc\s*ky|học\s*kỳ|semester)\s*(\d+)", re.IGNORECASE)
    cur_hit = cur_re.search(text)
    if cur_hit:
        constraints.current_semester_index = int(cur_hit.group(1))
        constraints.current_semester_source = "heuristic_current_hk"

    # Heuristic: completed semesters phrasing → next semester = max_completed + 1
    # Examples: "đã học xong HK1", "đã hoàn thành học kì 2", "xong semester 3".
    completed_hk_re = re.compile(
        r"(?:đã\s*(?:học\s*)?xong|da\s*(?:hoc\s*)?xong|đã\s*hoàn\s*thành|da\s*hoan\s*thanh|hoàn\s*tất|hoan\s*tat|xong)\s*(?:hk|học\s*kỳ|hoc\s*ky|học\s*kì|hoc\s*ki|semester)\s*(\d+)",
        re.IGNORECASE,
    )
    completed_hits = [int(m.group(1)) for m in completed_hk_re.finditer(text)]
    if completed_hits:
        # Store for debug/trace
        for k in completed_hits:
            if k not in (constraints.completed_semesters or []):
                (constraints.completed_semesters or []).append(k)
        # Only set from heuristic if we didn't already capture an explicit "đang học".
        if constraints.current_semester_index is None:
            constraints.current_semester_index = max(completed_hits) + 1
            constraints.current_semester_source = "heuristic_completed_hk"

    course_id_re = re.compile(r"\b([A-Z]{2,5}\d{2,4}[A-Z]{0,3})\b", re.IGNORECASE)
    for m in course_id_re.finditer(text):
        cid = m.group(1).upper().strip()
        if cid not in constraints.completed_courses:
            constraints.completed_courses.append(cid)

    # -------------------------
    # 2. LLM refinement (JSON)
    # -------------------------
    if not isinstance(cfg, dict):
        return constraints

    backend = str((cfg.get("llm") or {}).get("default_backend", "")).lower()
    if backend != "local":
        return constraints

    # IMPORTANT: LLM is allowed to override heuristic for current_semester_index
    # per user preference. We still keep heuristic-derived completed_semesters for logging.
    llm_prompt = (
        "Extract curriculum planning constraints from the user query.\n"
        "Rules:\n"
        "- If the user says they completed HKk / học kỳ k (đã học xong/đã hoàn thành/xong), then current_semester_index should be k+1.\n"
        "- If the user says they are currently in HKk (đang học), then current_semester_index should be k.\n"
        "- Extract course IDs like IT123IU, MA101IU, CS201, etc.\n"
        "- Output ONLY JSON that matches the provided schema.\n\n"
        f"USER QUERY:\n{question}\n"
    )

    try:
        data = llm_generate_json(
            prompt=llm_prompt,
            cfg=cfg,
            step="constraints",
            max_tokens=800,
        )
    except Exception as exc:
        logger.warning("parse_user_constraints: LLM refine failed: %s", exc)
        return constraints

    for key in [
        "min_credits_per_semester",
        "max_credits_per_semester",
        "terms_remaining",
    ]:
        val = data.get(key)
        if isinstance(val, int):
            setattr(constraints, key, val)

    # current_semester_index: LLM has priority if valid
    llm_cur = data.get("current_semester_index")
    if isinstance(llm_cur, int) and llm_cur > 0:
        if constraints.current_semester_index is not None and constraints.current_semester_index != llm_cur:
            logger.info(
                "parse_user_constraints: current_semester_index conflict heuristic=%s llm=%s (keeping LLM)",
                constraints.current_semester_index,
                llm_cur,
            )
        constraints.current_semester_index = llm_cur
        constraints.current_semester_source = "llm"

    # completed_semesters from LLM (optional, union with heuristic)
    llm_completed = data.get("completed_semesters")
    if isinstance(llm_completed, list):
        merged: List[int] = []
        seen = set()
        for v in (constraints.completed_semesters or []):
            if isinstance(v, int) and v not in seen:
                seen.add(v)
                merged.append(v)
        for v in llm_completed:
            if isinstance(v, int) and v not in seen:
                seen.add(v)
                merged.append(v)
        constraints.completed_semesters = merged

    # Merge courses lists defensively.
    def _merge_str_lists(a: List[str], b: Any) -> List[str]:
        out: List[str] = []
        seen2 = set()
        for x in (a or []):
            if isinstance(x, str) and x.strip():
                s = x.upper().strip()
                if s not in seen2:
                    seen2.add(s)
                    out.append(s)
        if isinstance(b, list):
            for x in b:
                if isinstance(x, str) and x.strip():
                    s = x.upper().strip()
                    if s not in seen2:
                        seen2.add(s)
                        out.append(s)
        return out

    constraints.completed_courses = _merge_str_lists(constraints.completed_courses or [], data.get("completed_courses"))
    constraints.preferred_courses = _merge_str_lists(constraints.preferred_courses or [], data.get("preferred_courses"))
    constraints.avoid_courses = _merge_str_lists(constraints.avoid_courses or [], data.get("avoid_courses"))

    return constraints


# =====================================================
# Slow planner orchestration (ILP)
# =====================================================

@log_call(level=logging.INFO, include_result=True)
def run_slow_planner_for_curriculum(
    question: str,
    cfg: Dict[str, Any],
    graph: Dict[str, Any],
    alias_normalizer: Optional[AliasNormalizer] = None,
    *,
    use_thesis_replacement: bool = False,
) -> Tuple[CurriculumPlan, Dict[str, Any]]:

    constraints = parse_user_constraints(question, cfg)

    # Normalize completed course IDs if an alias normalizer is available.
    if alias_normalizer and constraints.completed_courses:
        normed: List[str] = []
        for raw in constraints.completed_courses:
            try:
                resolved = alias_normalizer.resolve(raw)
                canonical = (resolved.get("canonical") or raw).upper().strip()
                if canonical and canonical not in normed:
                    normed.append(canonical)
            except Exception:
                # Keep raw token if normalization fails.
                token = raw.upper().strip()
                if token and token not in normed:
                    normed.append(token)
        constraints.completed_courses = normed

    total_years = int((cfg.get("curriculum") or {}).get("total_years", 4))
    semesters_per_year = int((cfg.get("curriculum") or {}).get("semesters_per_year", 2))

    # Infer current semester if not provided (conservative, KG-based).
    node_map = graph.get("node_map") or {}
    max_program_semesters = int(total_years * semesters_per_year)
    inferred_source = None
    if constraints.current_semester_index is None:
        inferred = None
        latest_possible = 0
        for cid in (constraints.completed_courses or []):
            node = node_map.get(cid)
            if not node:
                continue
            sem = getattr(node, "semester", None)
            if sem == "all":
                earliest = 1
            elif isinstance(sem, list) and sem:
                earliest = min(int(x) for x in sem if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit()))
            else:
                continue
            if earliest > latest_possible:
                latest_possible = earliest
        if latest_possible > 0:
            inferred = latest_possible + 1
            inferred_source = "kg_infer"
        if inferred is None:
            inferred = 1
            inferred_source = "default"
        # Clamp to [1..8] (ILP supports IU 1..8) and [1..max_program_semesters]
        inferred = max(1, min(int(inferred), min(8, max_program_semesters)))
        constraints.current_semester_index = inferred
        constraints.current_semester_source = inferred_source

    # Final clamp to ILP expected range [1..8]
    if constraints.current_semester_index is not None:
        constraints.current_semester_index = max(1, min(int(constraints.current_semester_index), 8))

    logger.info(
        "Parsed current_semester_index=%s source=%s | completed_courses=%d",
        constraints.current_semester_index,
        constraints.current_semester_source,
        len(constraints.completed_courses or []),
    )

    ilp_constraints = ILPUserCurriculumConstraints(
        total_years=total_years,
        semesters_per_year=semesters_per_year,
        min_credits_per_semester=int(constraints.min_credits_per_semester or 14),
        max_credits_per_semester=int(constraints.max_credits_per_semester or 24),
        terms_remaining=constraints.terms_remaining,
        current_semester_index=int(constraints.current_semester_index or 1),
        completed_courses=list(constraints.completed_courses or []),
        use_thesis_replacement=bool(use_thesis_replacement),
    )

    prereq_in = graph.get("prereq_in") or {}

    ilp_plan = _ilp_plan_curriculum_slow(
        node_map=node_map,
        prereq_in=prereq_in,
        user_constraints=ilp_constraints,
    )

    semesters_out: List[Dict[str, Any]] = []
    total_courses = 0

    for sem in ilp_plan.semesters or []:
        courses_payload: List[Dict[str, Any]] = []
        for c in sem.courses or []:
            courses_payload.append(
                {"id": c.id, "name": c.name or c.id, "credits": c.credits}
            )
        total_courses += len(courses_payload)
        semesters_out.append(
            {
                "semester": sem.index,
                "title": f"Học kỳ {sem.index}",
                "courses": courses_payload,
                "total_credits": sem.total_credits,
            }
        )

    plan = CurriculumPlan(
        semesters=semesters_out,
        summary={
            "status": ilp_plan.status,
            "message": ilp_plan.message,
            "semesters": len(semesters_out),
            "courses": total_courses,
        },
        meta={
            "solver_status": ilp_plan.status,
            "solver_message": ilp_plan.message,
            "total_years": total_years,
            "semesters_per_year": semesters_per_year,
        },
    )

    return plan, plan.meta
