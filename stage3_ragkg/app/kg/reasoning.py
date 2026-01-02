from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from app.logging_utils import get_logger, log_call
from app.kg.loader import CourseNode

logger = get_logger(__name__)


@log_call(level=logging.DEBUG, include_result=False)
def build_prereq_edges_evidence(
    *,
    plan_course_ids: List[str],
    prereq_in: Dict[str, List[str]],
    include_chain: bool = True,
    max_edges: int = 120,
) -> Dict[str, Any]:
    """
    Build a compact edge-only representation for prerequisite structure
    among the courses in a plan.

    Output:
      {
        "courses_considered": [...],
        "edges": ["COURSE <- PREREQ", ...],
        "truncated": bool (optional)
      }

    Notes:
      - This is intentionally token-light for LLM prompts.
      - If include_chain=True, includes prereqs-of-prereqs where available.
    """
    seeds = {c.strip() for c in (plan_course_ids or []) if isinstance(c, str) and c.strip()}
    if not seeds:
        return {"courses_considered": [], "edges": []}

    relevant: Set[str] = set(seeds)

    if include_chain:
        frontier: Set[str] = set(seeds)
        # bounded BFS to avoid runaway
        for _depth in range(5):
            nxt: Set[str] = set()
            for course in frontier:
                for pre in prereq_in.get(course, []) or []:
                    if pre not in relevant:
                        relevant.add(pre)
                        nxt.add(pre)
            if not nxt:
                break
            frontier = nxt

    edges: List[str] = []
    # Only include edges among relevant nodes
    for course in sorted(relevant):
        for pre in prereq_in.get(course, []) or []:
            if pre in relevant:
                edges.append(f"{course} <- {pre}")
                if len(edges) >= max_edges:
                    logger.warning(
                        "build_prereq_edges_evidence: reached max_edges=%d; truncating", max_edges
                    )
                    return {
                        "courses_considered": sorted(relevant),
                        "edges": edges,
                        "truncated": True,
                    }

    return {
        "courses_considered": sorted(relevant),
        "edges": edges,
    }


def _compact_course_node(node: CourseNode) -> Dict[str, Any]:
    """Return compact, token-aware course metadata for prompts."""
    name = None
    try:
        if isinstance(getattr(node, "name", None), dict):
            # Prefer English if present; fallback to any available.
            name = node.name.get("en") or node.name.get("vi")
            if not name and node.name:
                name = next(iter(node.name.values()))
    except Exception:
        name = None

    semesters = getattr(node, "semester", None)
    # semester can be "all" or list[int]

    credits_total = None
    try:
        credits_total = int(getattr(getattr(node, "credits", None), "total", None))
    except Exception:
        credits_total = None

    compulsory = getattr(node, "compulsory", None)
    is_elective = True if compulsory == "Elective" else False if compulsory == "Compulsory" else None

    out: Dict[str, Any] = {
        "id": getattr(node, "id", None),
        "name": name,
        "credits": credits_total,
        "semester": semesters,
        "compulsory": compulsory,
        "is_elective": is_elective,
    }

    # Only include these if they exist; keep small.
    try:
        if getattr(node, "area", None):
            out["area"] = list(node.area)[:5]
    except Exception:
        pass

    try:
        if getattr(node, "language", None):
            out["language"] = node.language
    except Exception:
        pass

    try:
        if getattr(node, "eligibility", None):
            el = node.eligibility
            out["eligibility"] = {
                "min_credits": getattr(el, "min_credits", None),
                "note": getattr(el, "note", None),
            }
    except Exception:
        pass

    # Remove None values
    return {k: v for k, v in out.items() if v is not None}


def _prereq_closure_limited(
    seeds: Set[str],
    prereq_in: Dict[str, List[str]],
    *,
    max_depth: int = 2,
) -> Set[str]:
    """Collect prerequisite closure up to max_depth hops (incoming)."""
    if not seeds:
        return set()

    visited: Set[str] = set(seeds)
    frontier: Set[str] = set(seeds)
    depth = 0

    while frontier and depth < max_depth:
        nxt: Set[str] = set()
        for c in frontier:
            for pre in prereq_in.get(c, []) or []:
                if pre not in visited:
                    visited.add(pre)
                    nxt.add(pre)
        frontier = nxt
        depth += 1

    return visited


@log_call(level=logging.INFO, include_result=False)
def build_relevant_kg_findings(
    *,
    seed_course_ids: List[str],
    node_map: Dict[str, CourseNode],
    prereq_in: Dict[str, List[str]],
    include_prereq_depth: int = 2,
    include_target_findings: bool = True,
    max_targets: int = 5,
    max_nodes: int = 120,
    max_edges: int = 400,
) -> Dict[str, Any]:
    """Build KG_FINDINGS with BOTH relevant nodes metadata + prereq edges.

    This keeps prompts compact by only including courses relevant to the current
    question/plan, plus a limited prerequisite closure.

    Output:
      {
        "meta": {...},
        "nodes": [...],
        "edges": [{"course": "X", "prereq": "Y"}, ...],
        "targets": [ optional per-target derived prereq fields ...]
      }
    """
    seeds = {str(x).strip() for x in (seed_course_ids or []) if str(x).strip()}
    if not seeds:
        return {"meta": {"seed_courses": [], "node_count": 0, "edge_count": 0}, "nodes": [], "edges": []}

    relevant = _prereq_closure_limited(seeds, prereq_in, max_depth=max(0, int(include_prereq_depth)))
    relevant_list = sorted(relevant)

    # Build edges among relevant courses (course <- prereq)
    edges: List[Dict[str, str]] = []
    for course in relevant_list:
        for pre in prereq_in.get(course, []) or []:
            if pre in relevant:
                edges.append({"course": course, "prereq": pre})
                if len(edges) >= max_edges:
                    break
        if len(edges) >= max_edges:
            break

    truncated_nodes = False
    truncated_edges = len(edges) >= max_edges

    # Nodes metadata
    nodes: List[Dict[str, Any]] = []
    for cid in relevant_list[: max_nodes]:
        node = node_map.get(cid)
        if node is None:
            continue
        nodes.append(_compact_course_node(node))
    if len(relevant_list) > max_nodes:
        truncated_nodes = True

    payload: Dict[str, Any] = {
        "meta": {
            "seed_courses": sorted(seeds),
            "include_prereq_depth": int(include_prereq_depth),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated_nodes": bool(truncated_nodes),
            "truncated_edges": bool(truncated_edges),
        },
        "nodes": nodes,
        "edges": edges,
    }

    if include_target_findings:
        targets: List[Dict[str, Any]] = []
        for cid in list(sorted(seeds))[: max_targets]:
            try:
                f = build_findings_for_course(cid, node_map=node_map, prereq_in=prereq_in)
                # Keep only deterministic, user-facing fields; drop debug-heavy fields.
                keep = {
                    "course": f.get("course"),
                    "required_courses": f.get("required_courses") or [],
                    "dependency_rules": f.get("dependency_rules") or [],
                    "prereq_required_by_level": f.get("prereq_required_by_level") or [],
                    "eligibility_note": f.get("eligibility_note"),
                }
                targets.append(keep)
            except Exception:
                logger.exception("build_relevant_kg_findings: target findings failed for %s", cid)
        if targets:
            payload["targets"] = targets

    # Metric-only KG observability (avoid logging full KG payload).
    try:
        preview_ids: List[str] = []
        for cid in payload.get("meta", {}).get("seed_courses", []) or []:
            s = str(cid).strip()
            if s and s not in preview_ids:
                preview_ids.append(s)
            if len(preview_ids) >= 20:
                break
        if not preview_ids:
            for n in (payload.get("nodes") or [])[:20]:
                if isinstance(n, dict) and n.get("id"):
                    preview_ids.append(str(n["id"]).strip())
        meta = payload.get("meta") or {}
        logger.debug(
            "KG_BUILD seeds=%d nodes=%d edges=%d truncated_nodes=%s truncated_edges=%s courses=%s capped=%d",
            len(meta.get("seed_courses") or []),
            int(meta.get("node_count", 0) or 0),
            int(meta.get("edge_count", 0) or 0),
            bool(meta.get("truncated_nodes", False)),
            bool(meta.get("truncated_edges", False)),
            preview_ids,
            20,
        )
    except Exception:
        # Never let logging break KG reasoning
        pass

    return payload


@log_call(level=logging.DEBUG, include_result=False)
def prereq_chain(course_id: str, prereq_in: Dict[str, List[str]]) -> List[List[str]]:
    """Return all prerequisite paths that end at `course_id` (root -> ... -> course_id)."""
    paths: List[List[str]] = []

    def dfs(cur: str, path: List[str], visited: Set[str]) -> None:
        incoming = prereq_in.get(cur, [])
        if not incoming:
            paths.append(path.copy())
            return
        for pre in incoming:
            if pre in visited:
                continue
            visited.add(pre)
            dfs(pre, [pre] + path, visited)
            visited.remove(pre)

    if course_id not in prereq_in:
        logger.debug("prereq_chain: course_id %s not found in prereq_in", course_id)
        return []

    dfs(course_id, [course_id], {course_id})
    logger.debug("prereq_chain: course_id=%s total_paths=%d", course_id, len(paths))
    return paths


@log_call(level=logging.DEBUG, include_result=False)
def levelize_from_chains(prereq_chains: List[List[str]]) -> Dict[str, Any]:
    """Convert prerequisite chains into a topological level ordering over the induced subgraph.

    Output:
      {
        "levels": [[...], [...], ...],
        "edges": [(pre, course), ...],
        "nodes": [...],
        "has_cycle": bool
      }
    """
    chains = [c for c in (prereq_chains or []) if isinstance(c, list) and c]
    if not chains:
        return {"levels": [], "edges": [], "nodes": [], "has_cycle": False}

    edges_set: Set[tuple[str, str]] = set()
    nodes_set: Set[str] = set()

    for chain in chains:
        for node in chain:
            nodes_set.add(str(node))
        for i in range(len(chain) - 1):
            a = str(chain[i]).strip()
            b = str(chain[i + 1]).strip()
            if a and b and a != b:
                edges_set.add((a, b))

    nodes = sorted(nodes_set)
    edges = sorted(edges_set)

    # Build adjacency + indegree
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    indeg: Dict[str, int] = {n: 0 for n in nodes}

    for a, b in edges:
        adj.setdefault(a, [])
        adj.setdefault(b, [])
        indeg.setdefault(a, 0)
        indeg.setdefault(b, 0)
        adj[a].append(b)
        indeg[b] += 1

    # Topological layering
    remaining: Set[str] = set(adj.keys())
    levels: List[List[str]] = []
    has_cycle = False

    while remaining:
        level = sorted([n for n in remaining if indeg.get(n, 0) == 0])
        if not level:
            has_cycle = True
            break
        levels.append(level)
        for n in level:
            remaining.remove(n)
            for m in adj.get(n, []):
                indeg[m] = max(0, indeg.get(m, 0) - 1)

    return {"levels": levels, "edges": edges, "nodes": nodes, "has_cycle": has_cycle}


@log_call(level=logging.DEBUG, include_result=False)
def course_detail(course_id: str, node_map: Dict[str, CourseNode]) -> Dict[str, Any]:
    node = node_map.get(course_id)
    if not node:
        return {"course": course_id, "error": "Course not found in KG"}

    # Prefer English name if present; fallback to any
    name = None
    if isinstance(node.name, dict):
        name = node.name.get("en") or node.name.get("vi")
        if not name and node.name:
            name = next(iter(node.name.values()))

    out: Dict[str, Any] = {
        "id": node.id,
        "name": name,
        "credits": {"total": node.credits.total, "theory": node.credits.theory, "lab": node.credits.lab},
        "area": node.area,
        "language": node.language,
        "semester": node.semester,
        "compulsory": node.compulsory,
        "source": node.source,
        "version": node.version,
    }

    # Optional eligibility metadata
    if getattr(node, "eligibility", None):
        out["eligibility"] = {
            "min_credits": node.eligibility.min_credits,
            "note": node.eligibility.note,
        }

    return out


def _dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


@log_call(level=logging.INFO, include_result=False)
def build_findings_for_course(
    course_id: str,
    node_map: Dict[str, CourseNode],
    prereq_in: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Build a structured set of findings for a single course.

    This function is used for LLM grounding and for debugging prereq logic.
    It returns multiple representations:
      - prereq_chains: all prerequisite paths (debug)
      - prereq_levels: topological layers induced by those paths
      - prereq_edges: edges in "A->B" direction for dependency descriptions
      - required_courses: AND-list of all prereqs required to take the course
      - dependency_rules: human-readable ordering constraints
      - parallel_groups: groups of courses that can be taken in parallel
    """
    course_id = (course_id or "").strip()
    if not course_id:
        return {"error": "Missing course_id"}

    if course_id not in node_map:
        return {"course": course_id, "error": "Course not found in KG"}

    # All chains ending at course_id (root -> ... -> course_id)
    chains = prereq_chain(course_id, prereq_in)

    topo = levelize_from_chains(chains)
    levels: List[List[str]] = topo.get("levels") or []

    # required_courses as an AND-set: all nodes in induced subgraph except the course itself
    required_courses: List[str] = []
    for lvl in levels:
        for c in lvl:
            if c != course_id:
                required_courses.append(c)
    required_courses = _dedupe_preserve_order(required_courses)

    # Build dependency_rules from edges
    dependency_rules: List[str] = []
    for a, b in (topo.get("edges") or []):
        dependency_rules.append(f"{a} học trước {b}")

    # Identify parallel groups per level
    parallel_groups: List[List[str]] = []
    for lvl in levels:
        group = [c for c in lvl if c != course_id]
        if len(group) >= 2:
            parallel_groups.append(group)

    prereq_required_by_level: List[Dict[str, Any]] = []
    for i, lvl in enumerate(levels, start=1):
        lvl_courses = [c for c in lvl if c != course_id]
        if lvl_courses:
            prereq_required_by_level.append({"level": i, "courses": lvl_courses})

    findings: Dict[str, Any] = {
        "course": course_id,
        "detail": course_detail(course_id, node_map),

        # Graph paths for explainability.
        "prereq_chains": chains,

        # Topological structure.
        "prereq_levels": levels,
        "prereq_edges": [f"{a}->{b}" for (a, b) in (topo.get("edges") or [])],
        "prereq_levels_has_cycle": bool(topo.get("has_cycle")),

        # AND-explicit prerequisite fields for user-facing answers.
        "required_courses": required_courses,
        "dependency_rules": dependency_rules,
        "parallel_groups": parallel_groups,
        "prereq_required_by_level": prereq_required_by_level,
    }

    node = node_map.get(course_id)
    if node is not None and getattr(node, "eligibility", None):
        try:
            findings["eligibility_note"] = {
                "min_credits": node.eligibility.min_credits,
                "note": node.eligibility.note,
                "source": getattr(node, "source", None),
                "version": getattr(node, "version", None),
            }
        except Exception:
            logger.exception(
                "build_findings_for_course: could not unpack eligibility metadata for %s",
                course_id,
            )
            findings["eligibility_note"] = {
                "error": "Could not unpack eligibility metadata for this course."
            }

    logger.debug(
        "build_findings_for_course: course_id=%s chains=%d required_courses=%d",
        course_id,
        len(findings.get("prereq_chains") or []),
        len(findings.get("required_courses") or []),
    )
    return findings


@log_call(level=logging.INFO, include_result=False)
def execute_kg_queries(
    kg_queries: List[Any],
    node_map: Dict[str, CourseNode],
    prereq_in: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Execute a list of KG queries and return results."""
    results: List[Dict[str, Any]] = []
    for q in kg_queries or []:
        try:
            op = (q.get("op") if isinstance(q, dict) else None) or ""
            course_id = (q.get("course_id") if isinstance(q, dict) else None) or ""
            op = str(op).strip()
            course_id = str(course_id).strip()

            if not op:
                results.append({"error": "Missing op in KG query", "query": q})
                continue
            if not course_id and op not in {"graph_stats"}:
                results.append({"op": op, "error": "Missing course_id in KG query", "query": q})
                continue

            if op == "prereq_chain":
                results.append(
                    {"op": op, "course_id": course_id, "result": prereq_chain(course_id, prereq_in)}
                )
            elif op == "course_detail":
                results.append(
                    {"op": op, "course_id": course_id, "result": course_detail(course_id, node_map)}
                )
            elif op in {"course_findings", "course_findings_full"}:
                results.append(
                    {
                        "op": op,
                        "course_id": course_id,
                        "result": build_findings_for_course(course_id, node_map, prereq_in),
                    }
                )
            elif op == "graph_stats":
                results.append(
                    {
                        "op": op,
                        "result": {
                            "num_courses": len(node_map or {}),
                            "num_edges": sum(len(v or []) for v in (prereq_in or {}).values()),
                        },
                    }
                )
            else:
                results.append({"op": op, "course_id": course_id, "error": "Unknown op", "query": q})
        except Exception as e:
            logger.exception("execute_kg_queries: failed for query=%s", q)
            results.append({"error": str(e), "query": q})

    return {"kg_results": results}
