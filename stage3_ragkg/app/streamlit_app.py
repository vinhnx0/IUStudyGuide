# app/streamlit_app.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from app.logging_utils import get_logger, setup_logging

try:
    st.set_page_config(
        page_title="IUStudyGuide — The Two Minds of a Chatbot",
        page_icon="🔎",
        layout="wide",
    )
except Exception:
    # Configure Streamlit page once when possible.
    pass

from app.rag.indexing import (
    IndexNotReadyError,
    ensure_vector_store_ready,
    reindex_all,
)
from app.rag.hybrid import HybridRetriever
from app.rag.alias_normalizer import AliasNormalizer
from app.router import detect_entities, decide_route
from app.llm import llm_generate_text, strip_think
from app.kg.loader import load_graph
from app.kg.reasoning import build_relevant_kg_findings
from app.guardrails import InputGuard, OutputGuard
from app.planner import (
    CurriculumPlan,
    decompose_question,
    run_slow_planner_for_curriculum,
)

logger = get_logger(__name__)


def load_config() -> Dict[str, Any]:
    """Load config.yaml from common locations."""
    import yaml

    for p in [Path("config.yaml"), Path("../config.yaml"), Path("/mnt/data/config.yaml")]:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("config.yaml not found at root/../ or /mnt/data/")


@st.cache_resource(show_spinner=False)
def get_cfg() -> Dict[str, Any]:
    """Cache parsed config and initialize logging once."""
    cfg = load_config()
    try:
        setup_logging(cfg)
    except Exception:
        logging.basicConfig(level=logging.INFO)
        logger.warning("Failed to fully apply setup_logging(cfg); using basicConfig fallback")
    logger.info("Configuration loaded and logging initialized.")
    return cfg


@st.cache_resource(show_spinner=False)
def get_retriever(cfg: Dict[str, Any]) -> HybridRetriever:
    """Create a HybridRetriever instance."""
    return HybridRetriever(cfg)


@st.cache_resource(show_spinner=False)
def get_alias_normalizer(cfg: Dict[str, Any]) -> AliasNormalizer:
    """Create an AliasNormalizer instance."""
    return AliasNormalizer(cfg)


@st.cache_resource(show_spinner=False)
def get_graph(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load KG graph."""
    kg_path = cfg.get("paths", {}).get("kg_json")
    g, node_map, prereq_in, prereq_out = load_graph(kg_path)
    return {
        "graph": g,
        "node_map": node_map,
        "prereq_in": prereq_in,
        "prereq_out": prereq_out,
    }


def _load_prompt(path_candidates: List[Path]) -> str:
    """Read a prompt from the first existing path candidate."""
    for p in path_candidates:
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found: tried {path_candidates}")


def load_answer_synth_prompt() -> str:
    """Load prompt used for FAST synthesis."""
    return _load_prompt(
        [
            Path("app/prompts/answer_synth_prompt.txt"),
            Path("prompts/answer_synth_prompt.txt"),
            Path("answer_synth_prompt.txt"),
        ]
    )


def load_slow_planning_synth_prompt() -> str:
    """Load prompt used for SLOW planning synthesis."""
    return _load_prompt(
        [
            Path("app/prompts/slow_planning_synth_prompt.txt"),
            Path("prompts/slow_planning_synth_prompt.txt"),
            Path("slow_planning_synth_prompt.txt"),
        ]
    )


_TEMPLATE_BEGIN_PREFIX = "===TEMPLATE_BEGIN:"


def get_prompt_prefix(full_prompt: str) -> str:
    # Return everything before the first occurrence of a template marker.
    full_prompt = full_prompt or ""
    i = full_prompt.find(_TEMPLATE_BEGIN_PREFIX)
    if i == -1:
        return full_prompt
    return full_prompt[:i]


def slice_template_block(full_prompt: str, template_id: str) -> str:
    # Return ONLY the template block for the given template_id.
    # If markers are missing/not found, return full_prompt (fallback) and log a warning.
    full_prompt = full_prompt or ""
    template_id = (template_id or "").strip()
    begin = f"===TEMPLATE_BEGIN:{template_id}==="
    end = f"===TEMPLATE_END:{template_id}==="

    if _TEMPLATE_BEGIN_PREFIX not in full_prompt:
        logger.warning("Prompt slicing markers not found; falling back to full prompt")
        return full_prompt

    a = full_prompt.find(begin)
    if a == -1:
        logger.warning("Prompt template block not found; falling back to full prompt | template_id=%s", template_id)
        return full_prompt

    b = full_prompt.find(end, a + len(begin))
    if b == -1:
        logger.warning("Prompt template end marker missing; falling back to full prompt | template_id=%s", template_id)
        return full_prompt

    return full_prompt[a + len(begin) : b].strip("\n")


def _select_answer_template(intent: str) -> str:
    """Map router intent -> answer template id for FAST synthesis.

    Planning (course_planning) is handled by ILP/UI; no LLM synthesis is required.
    """
    i = (intent or "").strip().lower()
    if i == "course_lookup":
        return "LOOKUP_V1"
    if i == "prerequisite_reasoning":
        return "PREREQ_PATH_V1"
    if i == "eligibility_checking":
        return "ELIGIBILITY_CHECKLIST_V1"
    return "GENERAL_QNA_V1"


def format_retrieval_evidence(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a string block for the LLM prompt."""
    lines: List[str] = []
    for i, c in enumerate(chunks or [], start=1):
        text = (c.get("text") or "").strip()
        meta = c.get("meta") or c.get("metadata") or {}
        source = meta.get("source", meta.get("title", "unknown"))
        score = c.get("score", None)
        score_str = "" if score is None else f"(score={score:.3f}) "
        lines.append(f"[{i}] {score_str}{source}\n{text}")
    return "\n\n".join(lines).strip()


def _safe_len(x) -> int:
    """Safe length helper for logging prompt sizes."""
    if x is None:
        return 0
    if isinstance(x, str):
        return len(x)
    try:
        return len(str(x))
    except Exception:
        return 0


def _capped_course_preview(values: Any, *, cap: int = 20) -> List[str]:
    """Return a stable, capped list of course IDs for logs.

    This keeps logs small and avoids dumping large KG structures.
    """
    ids: List[str] = []
    try:
        for v in (values or []):
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            if s not in ids:
                ids.append(s)
            if len(ids) >= int(cap):
                break
    except Exception:
        return []
    return ids


def _precompute_prereq_ordering_line(kg_payload: Dict[str, Any], target: str) -> str:
    """
    Deterministically build the minimal ordering line for PREREQ from:
    KG_FINDINGS.targets[*].prereq_required_by_level

    Output example:
    "Thứ tự tối thiểu gợi ý: IT149IU, MA001IU → IT069IU, IT154IU → IT172IU"
    """
    if not target:
        return ""

    targets = (kg_payload or {}).get("targets") or []
    tmatch = None
    for t in targets:
        if isinstance(t, dict) and str(t.get("course", "")).strip().upper() == str(target).strip().upper():
            tmatch = t
            break
    if not tmatch:
        return ""

    lvls = tmatch.get("prereq_required_by_level") or []
    if not isinstance(lvls, list) or not lvls:
        return ""

    # sort by "level" if present; otherwise keep given order
    def _lvl_key(x):
        try:
            return int(x.get("level", 10**9))
        except Exception:
            return 10**9

    parts: List[str] = []
    for item in sorted([x for x in lvls if isinstance(x, dict)], key=_lvl_key):
        courses = item.get("courses") or []
        # stable de-dup, preserve order
        seen = set()
        cleaned = []
        for c in courses:
            s = str(c).strip().upper()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        if cleaned:
            parts.append(", ".join(cleaned))

    if not parts:
        return ""

    tgt = str(target).strip().upper()
    chain = " → ".join(parts + [tgt])
    return f"Thứ tự tối thiểu gợi ý: {chain}"


def render_prereq_path_v1(kg_payload: dict, target_course: str) -> str:
    """Deterministically render PREREQ_PATH_V1 answer (no LLM).

    Enforces EXACTLY 4 paragraphs separated by EXACTLY ONE blank line.
    """
    tgt = (target_course or "").strip().upper()
    if not tgt:
        return ""

    payload = kg_payload or {}
    targets = payload.get("targets") or []

    tmatch = None
    for t in targets:
        if isinstance(t, dict) and str(t.get("course", "")).strip().upper() == tgt:
            tmatch = t
            break

    # Build node name map for fallbacks
    node_name: dict[str, str] = {}
    for n in payload.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        cid = str(n.get("id", "")).strip().upper()
        if not cid:
            continue
        node_name[cid] = (n.get("name") or "").strip()

    course_name = ""
    course_display = ""
    required_display: list[str] = []
    required_codes: list[str] = []

    if isinstance(tmatch, dict):
        course_name = (tmatch.get("course_name") or "").strip()
        course_display = (tmatch.get("course_display") or "").strip()
        required_display = [
            str(x).strip()
            for x in (tmatch.get("required_courses_display") or [])
            if str(x).strip()
        ]
        required_codes = [
            str(x).strip().upper()
            for x in (tmatch.get("required_courses") or [])
            if str(x).strip()
        ]

    if not course_display:
        nm = course_name or node_name.get(tgt, "")
        course_display = f"{tgt} ({nm})" if nm else tgt

    # Normalize required_display; if missing, build from required_codes + node names
    cleaned_required_display: list[str] = []
    if required_display:
        for s in required_display:
            if s and s not in cleaned_required_display:
                cleaned_required_display.append(s)
    else:
        seen: set[str] = set()
        for cc in required_codes:
            if not cc or cc in seen:
                continue
            seen.add(cc)
            nm = node_name.get(cc, "")
            cleaned_required_display.append(f"{cc} ({nm})" if nm else cc)

    # P3 ordering line: deterministic precompute (levels)
    ordering_line = _precompute_prereq_ordering_line(payload, tgt)

    has_prereqs = bool(cleaned_required_display)

    if not has_prereqs:
        # No prerequisites
        p1 = f"{course_display} không yêu cầu môn tiên quyết."
        p2 = "Danh sách tiên quyết (bắt buộc): Không có"
        p3 = f"Thứ tự tối thiểu gợi ý: Bạn có thể đăng ký {tgt} ngay (theo đúng lộ trình/điều kiện đăng ký của chương trình)."
    else:
        # Has prerequisites
        p1 = f"Để học được {course_display}, bạn cần hoàn thành các môn tiên quyết sau."
        p2 = "Danh sách tiên quyết (bắt buộc): " + ", ".join(cleaned_required_display)
        if not ordering_line:
            ordering_line = f"Thứ tự tối thiểu gợi ý: Hoàn thành tất cả các môn tiên quyết trước khi đăng ký {tgt}."
        p3 = ordering_line

    p4 = "Citations: [Curriculum Knowledge Graph]"

    return "\n\n".join([p1, p2, p3, p4]).strip() + "\n"


def _log_kg_metrics(*, tag: str, kg_payload: Dict[str, Any], findings_json: str, req_id: Any = None) -> None:
    """Log KG_FINDINGS metrics (no content)."""
    meta = (kg_payload or {}).get("meta") or {}
    node_count = int(meta.get("node_count", 0) or 0)
    edge_count = int(meta.get("edge_count", 0) or 0)
    seeds = _capped_course_preview(meta.get("seed_courses") or [])

    # Optional extra preview from included nodes (capped). Keep deterministic.
    try:
        node_ids = _capped_course_preview([n.get("id") for n in (kg_payload.get("nodes") or []) if isinstance(n, dict)])
    except Exception:
        node_ids = []

    preview = seeds or node_ids
    logger.info(
        "KG_FINDINGS tag=%s req_id=%s chars=%d nodes=%d edges=%d courses=%s capped=%d",
        tag,
        str(req_id) if req_id is not None else "-",
        len(findings_json or ""),
        node_count,
        edge_count,
        preview,
        20,
    )


def _log_kg_preview(*, tag: str, kg_payload: Dict[str, Any], req_id: Any = None,
                    max_nodes: int = 40, max_edges: int = 80) -> None:
    """Debug-only: log a small preview of nodes/edges to verify KG extraction."""
    try:
        nodes = kg_payload.get("nodes") or []
        edges = kg_payload.get("edges") or []
        targets = kg_payload.get("targets") or []

        node_ids = []
        for n in nodes:
            if isinstance(n, dict) and n.get("id"):
                node_ids.append(str(n["id"]).strip())
            if len(node_ids) >= max_nodes:
                break

        edge_pairs = []
        for e in edges:
            if isinstance(e, dict) and e.get("course") and e.get("prereq"):
                edge_pairs.append(f'{e["course"]} <- {e["prereq"]}')
            if len(edge_pairs) >= max_edges:
                break

        target_summ = []
        for t in targets:
            if not isinstance(t, dict):
                continue
            course = t.get("course")
            reqs = t.get("required_courses") or []
            lvls = t.get("prereq_required_by_level") or []
            target_summ.append(
                {
                    "course": course,
                    "required_courses": reqs,
                    "prereq_required_by_level": lvls,
                }
            )

        logger.info(
            "KG_PREVIEW tag=%s req_id=%s nodes=%d edges=%d node_ids=%s edge_pairs=%s targets=%s",
            tag,
            str(req_id) if req_id is not None else "-",
            len(nodes),
            len(edges),
            node_ids,
            edge_pairs,
            target_summ,
        )
    except Exception:
        logger.exception("KG_PREVIEW failed")


def sanitize_kg_findings_for_llm(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove debug-heavy fields from KG findings used in LLM prompts."""
    out: List[Dict[str, Any]] = []
    for f in findings or []:
        if not isinstance(f, dict):
            continue
        ff = dict(f)
        ff.pop("prereq_chains", None)
        out.append(ff)
    return out


def show_sidebar(cfg: Dict[str, Any], index_status: Dict[str, Any]) -> None:
    """Render sidebar configuration & info."""
    st.sidebar.title("Stage-3 RAG + KG + ILP Planner")
    st.sidebar.markdown(
        """
This demo combines:
- Hybrid retrieval (BM25 + Dense/Qdrant)
- Knowledge Graph reasoning
- Optional ILP-based curriculum planning
        """.strip()
    )

    st.sidebar.subheader("Index status")
    st.sidebar.write(f"Collection: `{index_status.get('collection_name', '')}`")
    st.sidebar.write(f"Qdrant points: `{index_status.get('qdrant_points', 0)}`")
    st.sidebar.write(f"BM25 ok: `{index_status.get('bm25_ok', False)}`")
    st.sidebar.write(f"Vector dim: `{index_status.get('vector_dim', '')}`")

    st.sidebar.subheader("LLM Backend")
    llm_cfg = cfg.get("llm", {}) or {}
    st.sidebar.write(f"Default backend: `{llm_cfg.get('default_backend', 'N/A')}`")

    st.sidebar.subheader("Logging")
    log_cfg = cfg.get("logging", {}) or {}
    st.sidebar.write(f"Level: `{log_cfg.get('level', 'INFO')}`")
    st.sidebar.write(f"To file: `{log_cfg.get('to_file', False)}`")
    if log_cfg.get("to_file", False):
        st.sidebar.write(f"Filename: `{log_cfg.get('filename', '')}`")


def render_plan(plan: CurriculumPlan) -> None:
    """
    Render curriculum plan to Streamlit UI.

    UX rules:
    - If INFEASIBLE: show a clear user-facing message + general suggestions (Vietnamese only)
    - If FEASIBLE/OPTIMAL: render by semester, one course per line
    """

    # ---------- INFEASIBLE ----------
    solver_status = (plan.summary or {}).get("status") or (plan.meta or {}).get("solver_status")

    if solver_status == "INFEASIBLE":
        st.error("Không tìm được kế hoạch học phù hợp (infeasible).")

        st.markdown(
            """
            Bạn có thể thử:
            - Tăng số **tín chỉ tối đa mỗi học kỳ** hoặc giảm **tín chỉ tối thiểu mỗi học kỳ** (nếu có)
            - Tăng **số học kỳ / số năm** trong kế hoạch
            - Nới lỏng một số **ràng buộc cố định** (ví dụ: giới hạn môn học được chọn, hoặc học kỳ bắt đầu lên kế hoạch)
            - **Kiểm tra lại danh sách các môn đã hoàn thành**
            """
        )

        logger.info("Rendered INFEASIBLE plan to UI")
        return

    # ---------- ERROR ----------
    solver_status = (plan.summary or {}).get("status") or (plan.meta or {}).get("solver_status")

    if solver_status == "ERROR":
        st.error("Đã xảy ra lỗi hệ thống khi lập kế hoạch.")
        solver_message = (plan.summary or {}).get("message") or (plan.meta or {}).get("solver_message")

        if solver_message:
            st.caption(solver_message)

        logger.error("Rendered ERROR plan to UI")
        return

    # ---------- FEASIBLE / OPTIMAL ----------
    if not plan.semesters:
        st.warning("Không có dữ liệu kế hoạch để hiển thị.")
        logger.warning("Plan has no semesters but status is not INFEASIBLE/ERROR")
        return

    for sem in plan.semesters:
        semester_label = sem.get("semester", "")
        total_credits = sem.get("total_credits", 0)

        st.subheader(f"Semester {semester_label}")
        st.caption(f"Total: {total_credits} credits")

        courses = sem.get("courses", [])
        if not courses:
            st.write("_Không có môn học trong học kỳ này._")
            continue

        # Stable ordering: by course_id
        courses_sorted = sorted(courses, key=lambda c: c.get("id", ""))

        for c in courses_sorted:
            cid = c.get("id", "")
            name = c.get("name", cid)
            credits = c.get("credits", None)

            if credits is not None:
                st.write(f"{cid} — {name} — {credits} credits")
            else:
                st.write(f"{cid} — {name}")

    logger.info("Rendered FEASIBLE/OPTIMAL plan to UI")


def _render_index_rebuild_ui(cfg: Dict[str, Any], status: Dict[str, Any]) -> None:
    st.warning("Index is missing or empty. Build/rebuild the index to enable retrieval.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        build_clicked = st.button("Build index", use_container_width=True)
    with col_b:
        rebuild_clicked = st.button("Rebuild from scratch", use_container_width=True)

    if build_clicked or rebuild_clicked:
        force = bool(rebuild_clicked)
        label = "Rebuilding from scratch..." if force else "Building index..."

        with st.status(label, expanded=True) as s:
            try:
                stats = reindex_all(cfg, force_recreate=force)
                s.update(label="Index build complete", state="complete")
                st.success("Index ready.")
                st.json(stats)

                # Clear cached resources so retriever/BM25 reload the fresh artifacts.
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass

                st.rerun()
            except Exception as exc:
                logger.exception("Index build failed")
                s.update(label="Index build failed", state="error")
                st.error(f"Index build failed: {exc}")

    st.stop()


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _chunk_meta(chunk: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(chunk, dict):
        return {}
    meta = chunk.get("meta") or chunk.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def render_evidence_panels(*, rag_chunks: List[Dict[str, Any]] | None, kg_payload: Dict[str, Any] | None) -> None:
    # -------------------------
    # Evidence (RAG)
    # -------------------------
    rag_chunks = rag_chunks or []
    with st.expander(f"Evidence (RAG) — top {len(rag_chunks)} chunks", expanded=False):
        if not rag_chunks:
            st.caption("No retrieval evidence captured for this request.")
        else:
            for i, ch in enumerate(rag_chunks, start=1):
                meta = _chunk_meta(ch)
                title = (meta.get("title") or meta.get("source_title") or meta.get("doc_title") or "").strip()
                source = (meta.get("source") or meta.get("path") or meta.get("doc_id") or meta.get("id") or "").strip()
                score = _safe_float(ch.get("score", None), default=_safe_float(meta.get("score", None), 0.0))
                text = (ch.get("text") or ch.get("content") or "").strip()

                label = title or source or f"chunk_{i}"
                st.markdown(f"**#{i}. {label}**  \n`score={score:.4f}`")
                if source and title:
                    st.caption(f"source: {source}")
                elif source:
                    st.caption(f"source: {source}")

                # tránh UI quá dài
                if len(text) > 2500:
                    st.code(text[:2500] + "\n...\n[TRUNCATED]", language="text")
                else:
                    st.code(text, language="text")

                st.divider()

    # -------------------------
    # Evidence (KG)
    # -------------------------
    with st.expander("Evidence (KG)", expanded=False):
        if not kg_payload:
            st.caption("No KG evidence captured for this request.")
            return

        meta = (kg_payload.get("meta") or {}) if isinstance(kg_payload, dict) else {}
        if isinstance(meta, dict) and meta:
            node_count = meta.get("node_count", 0)
            edge_count = meta.get("edge_count", 0)
            truncated_nodes = meta.get("truncated_nodes", False)
            truncated_edges = meta.get("truncated_edges", False)
            seeds = meta.get("seed_courses", [])
            st.caption(
                f"nodes={node_count} | edges={edge_count} | "
                f"truncated_nodes={bool(truncated_nodes)} | truncated_edges={bool(truncated_edges)} | "
                f"seed_courses={seeds}"
            )

        targets = kg_payload.get("targets") or []
        edges = kg_payload.get("edges") or []
        nodes = kg_payload.get("nodes") or []

        # Targets: "enough" detail to verify
        st.markdown("### Targets")
        if not targets:
            st.caption("No targets extracted.")
        else:
            for t in targets[:8]:
                if not isinstance(t, dict):
                    continue
                course = t.get("course_display") or t.get("course") or ""
                reqs = t.get("required_courses_display") or t.get("required_courses") or []
                lvls = t.get("prereq_required_by_level_display") or t.get("prereq_required_by_level") or []
                st.markdown(f"- **{course}**")
                if reqs:
                    st.markdown(f"  - required: {', '.join(reqs) if isinstance(reqs, list) else str(reqs)}")

                if lvls:
                    # lvls can be:
                    # - list[str] (rare)
                    # - list[dict] like {"level": 1, "courses_display": ["A (..)", "B(..)"]}
                    # - anything else
                    if isinstance(lvls, list):
                        parts = []
                        for item in lvls:
                            if isinstance(item, str):
                                s = item.strip()
                                if s:
                                    parts.append(s)
                                continue

                            if isinstance(item, dict):
                                lvl = item.get("level")
                                cds = item.get("courses_display") or item.get("courses") or []
                                if isinstance(cds, list):
                                    cds_str = ", ".join([str(x) for x in cds if str(x).strip()])
                                else:
                                    cds_str = str(cds)

                                if cds_str.strip():
                                    parts.append(f"Level {lvl}: {cds_str}" if lvl is not None else cds_str)
                                continue

                            # fallback for odd types
                            parts.append(str(item))

                        st.markdown("  - by level: " + " | ".join(parts))
                    else:
                        st.markdown("  - by level: " + str(lvls))

        # Edges: course <- prereq
        st.markdown("### Edges (course ⇐ prereq)")
        if edges:
            preview_edges = []
            for e in edges[:80]:
                if not isinstance(e, dict):
                    continue
                preview_edges.append(
                    {"course": e.get("course", ""), "prereq": e.get("prereq", "")}
                )
            st.dataframe(preview_edges, use_container_width=True, hide_index=True)
        else:
            st.caption("No edges extracted.")

        # Nodes: only key fields
        st.markdown("### Nodes (preview)")
        if nodes:
            preview_nodes = []
            for n in nodes[:60]:
                if not isinstance(n, dict):
                    continue
                preview_nodes.append(
                    {
                        "id": n.get("id", ""),
                        "name": n.get("name", ""),
                        "semester": n.get("semester", ""),
                        "credits": n.get("credits", ""),
                        "is_elective": n.get("is_elective", ""),
                    }
                )
            st.dataframe(preview_nodes, use_container_width=True, hide_index=True)
        else:
            st.caption("No nodes extracted.")


def main() -> None:
    cfg = get_cfg()
    import app.llm as _llm
    logger.info("RUNTIME CHECK: app.llm file=%s", getattr(_llm, "__file__", "unknown"))

    # Always validate index status early so the app never crashes after DB reset.
    with st.spinner("Checking index status..."):
        index_status = ensure_vector_store_ready(cfg)

    show_sidebar(cfg, index_status)

    # If missing/empty, offer rebuild UI and halt before creating retriever.
    if index_status.get("needs_reindex", False):
        st.title("IUStudyGuide — The Two Minds of a Chatbot")
        _render_index_rebuild_ui(cfg, index_status)
        return

    input_guard = InputGuard()
    output_guard = OutputGuard()

    # Only create retriever once indexes are confirmed ready.
    retriever = get_retriever(cfg)
    alias_normalizer = get_alias_normalizer(cfg)
    graph = get_graph(cfg)
    node_map = graph.get("node_map") or {}
    prereq_in = graph.get("prereq_in") or {}

    st.title("IUStudyGuide — The Two Minds of a Chatbot")

    # Session state init
    st.session_state.setdefault("planning_active", False)
    st.session_state.setdefault("planning_question", "")
    st.session_state.setdefault("planning_decision", None)
    st.session_state.setdefault("planning_entities", [])
    st.session_state.setdefault("planning_decomp", None)
    st.session_state.setdefault("planning_retrieval_scores", [])

    st.session_state.setdefault("last_route_decision", None)
    st.session_state.setdefault("last_plan", None)
    st.session_state.setdefault("last_plan_meta", {})
    st.session_state.setdefault("last_answer", "")

    # Evidence snapshots for UI
    st.session_state.setdefault("last_rag_chunks", [])
    st.session_state.setdefault("last_kg_payload_fast", None)
    st.session_state.setdefault("last_kg_payload_slow", None)
    # Basic request timing (end-to-end runtime per user submission)
    st.session_state.setdefault("req_id", None)
    st.session_state.setdefault("req_start_time", None)
    st.session_state.setdefault("req_logged", True)

    question = st.text_area(
        "Ask a question about IU courses / curriculum:",
        height=120,
        placeholder="e.g., 'What are prerequisites for IT151IU?' or 'Plan my next 3 semesters with 18 credits each.'",
    )

    use_llm_router = True
    fast_threshold = 0.7

    with st.form("run_form", clear_on_submit=False):
        submitted_run = st.form_submit_button("Run")

    if submitted_run:
        # Start timing at request receipt (FAST or SLOW routing decision happens after this point)
        st.session_state.req_id = time.time_ns()
        st.session_state.req_start_time = time.perf_counter()
        st.session_state.req_logged = False

        guard_res = input_guard.check(question)
        if not guard_res.get("ok", True):
            st.error("Input rejected by guardrails.")
            return

        question_clean = guard_res.get("clean_text", question)
        if not question_clean:
            st.warning("Please enter a question.")
            return

        # Reset outputs from previous run   
        st.session_state.last_question = question_clean
        st.session_state.last_plan = None
        st.session_state.last_plan_meta = {}
        st.session_state.last_answer = ""
        st.session_state.last_rag_chunks = []
        st.session_state.last_kg_payload = None

        entities = detect_entities(question_clean, alias_normalizer=alias_normalizer)

        decomp = decompose_question(question_clean, cfg)
        decomp_entities = decomp.get("entities") or []
        decomp_sub_questions = decomp.get("sub_questions") or []
        kg_queries = decomp.get("kg_queries") or []

        try:
            chunks = retriever.retrieve(question_clean, top_k=3)
        except IndexNotReadyError as exc:
            st.error(str(exc))
            _render_index_rebuild_ui(cfg, index_status)
            return

        retrieval_scores = [float(c.get("score", 0.0)) for c in chunks]
        evidence_block = format_retrieval_evidence(chunks)

        decision = decide_route(
            question_clean,
            retrieval_scores=retrieval_scores,
            entities=entities,
            use_llm_router=use_llm_router,
            fast_threshold=fast_threshold,
            cfg=cfg,
            decomp_entities=decomp_entities,
            kg_queries=kg_queries,
            decomp_sub_questions=decomp_sub_questions,
        )

        st.session_state.last_route_decision = asdict(decision)
        st.session_state.last_rag_chunks = list(chunks or [])
        is_planning = decision.route == "SLOW" and decision.intent == "course_planning"
        st.session_state.planning_active = bool(is_planning)
        st.session_state.planning_question = question_clean
        st.session_state.planning_decision = asdict(decision)
        st.session_state.planning_entities = list(decision.entities or [])
        st.session_state.planning_decomp = decomp
        st.session_state.planning_retrieval_scores = retrieval_scores

        if not is_planning:
            # Build compact KG_FINDINGS: relevant nodes + edges (+ derived per-target fields).
            kg_payload = build_relevant_kg_findings(
                seed_course_ids=list(decision.entities or []),
                node_map=node_map,
                prereq_in=prereq_in,
                include_prereq_depth=2,
                include_target_findings=True,
            )
            st.session_state.last_kg_payload = kg_payload
            findings_json = json.dumps(kg_payload, ensure_ascii=False, indent=2)

            _log_kg_metrics(
                tag="FAST",
                kg_payload=kg_payload,
                findings_json=findings_json,
                req_id=st.session_state.get("req_id"),
            )
            _log_kg_preview(
                tag="FAST",
                kg_payload=kg_payload,
                req_id=st.session_state.get("req_id"),
            )
            synth_template = load_answer_synth_prompt()

            template_id = _select_answer_template(decision.intent)
            logger.info(
                "Answer synthesis | intent=%s template_id=%s route=%s",
                decision.intent,
                template_id,
                decision.route,
            )
            # Default empty unless PREREQ template needs it
            precomputed_prereq_ordering = ""

            if template_id == "PREREQ_PATH_V1":
                # choose target deterministically (prefer router decision)
                target_course = ""
                for lst in [list(decision.entities or []), list(decomp_entities or []), list(entities or [])]:
                    if lst:
                        target_course = str(lst[0]).strip().upper()
                        break

                precomputed_prereq_ordering = _precompute_prereq_ordering_line(kg_payload, target_course)
                logger.info(
                    "PREREQ_ORDERING_PRECOMPUTED req_id=%s target=%s line=%s",
                    str(st.session_state.get("req_id", "-")),
                    target_course,
                    precomputed_prereq_ordering or "(empty)",
                )

            if template_id == "PREREQ_PATH_V1":
                answer = render_prereq_path_v1(kg_payload, target_course)
                out_res = output_guard.check(answer)
                if not out_res.get("ok", True):
                    st.error("Output rejected by guardrails.")
                    st.session_state.last_answer = ""
                else:
                    st.session_state.last_answer = answer

                # End timing right before the UI renders the final answer.
                if not st.session_state.get("req_logged", True):
                    start_t = st.session_state.get("req_start_time")
                    if isinstance(start_t, (int, float)):
                        elapsed = time.perf_counter() - float(start_t)
                        logger.info("Total request runtime: %.3f seconds", elapsed)
                    st.session_state.req_logged = True

            else:
                prompt = synth_template.format(
                    template_id=template_id,
                    evidence=evidence_block,
                    question=question_clean,
                    entities=json.dumps(decision.entities or [], ensure_ascii=False),
                    intent=decision.intent,
                    route=decision.route,
                    confidence=f"{decision.confidence:.3f}",
                    signals=", ".join(decision.signals or []),
                    kg_findings=findings_json,
                    curriculum_plan="",
                    precomputed_prereq_ordering=precomputed_prereq_ordering,
                )

            if template_id != "PREREQ_PATH_V1":
                logger.debug(
                    "LLM prompt breakdown | template=%d question=%d evidence=%d kg=%d TOTAL=%d",
                    _safe_len(synth_template),
                    _safe_len(question_clean),
                    _safe_len(evidence_block),
                    _safe_len(findings_json),
                    _safe_len(prompt),
                )

                answer = llm_generate_text(prompt, cfg, caller="synthesis")
                answer = strip_think(answer)

                out_res = output_guard.check(answer)
                if not out_res.get("ok", True):
                    st.error("Output rejected by guardrails.")
                    st.session_state.last_answer = ""
                else:
                    st.session_state.last_answer = answer

                # End timing right before the UI renders the final answer.
                if not st.session_state.get("req_logged", True):
                    start_t = st.session_state.get("req_start_time")
                    if isinstance(start_t, (int, float)):
                        elapsed = time.perf_counter() - float(start_t)
                        logger.info("Total request runtime: %.3f seconds", elapsed)
                    st.session_state.req_logged = True

    # ----------------------------
    # SLOW planning UI
    # ----------------------------
    if st.session_state.planning_active:
        st.subheader("Planning Track")

        track_choice = st.radio(
            "Choose a thesis option for this plan:",
            options=[
                "Plan with Thesis (IT058IU)",
                "Replace Thesis with IT168IU + 2 electives",
            ],
            index=0,
            horizontal=True,
            key="planning_track_choice",
        )

        use_thesis_replacement = bool(track_choice.startswith("Replace"))

        with st.form("planning_form", clear_on_submit=False):
            submitted_generate = st.form_submit_button("Generate plan")

        if submitted_generate:
            # Start timing at request receipt for planning generation.
            st.session_state.req_id = time.time_ns()
            st.session_state.req_start_time = time.perf_counter()
            st.session_state.req_logged = False

            q = st.session_state.planning_question
            decision_dict = st.session_state.planning_decision or {}
            entities_planning = st.session_state.planning_entities or []

            with st.spinner("Running ILP planner..."):
                try:
                    curriculum_plan, planner_meta = run_slow_planner_for_curriculum(
                        q,
                        cfg=cfg,
                        graph=graph,
                        alias_normalizer=alias_normalizer,
                        use_thesis_replacement=use_thesis_replacement,
                    )
                except Exception as exc:
                    st.error(f"ILP planner failed: {exc}")
                    curriculum_plan = None
                    planner_meta = {"error": str(exc)}

            st.session_state.last_plan = curriculum_plan
            st.session_state.last_plan_meta = planner_meta

            # Build KG evidence for plan (compact, no raw dump)
            findings_json = "(no plan)"
            curriculum_plan_str = ""
            if curriculum_plan is not None:
                curriculum_plan_str = json.dumps(asdict(curriculum_plan), ensure_ascii=False, indent=2)

                try:
                    plan_course_ids: List[str] = []
                    for sem in (curriculum_plan.semesters or []):
                        for c in (sem.get("courses") or []):
                            if isinstance(c, dict) and c.get("id"):
                                plan_course_ids.append(str(c["id"]).strip())
                            elif isinstance(c, str):
                                plan_course_ids.append(c.strip())

                    kg_payload = build_relevant_kg_findings(
                        seed_course_ids=plan_course_ids,
                        node_map=node_map,
                        prereq_in=prereq_in,
                        include_prereq_depth=3,
                        include_target_findings=False,
                    )
                    findings_json = json.dumps(kg_payload, ensure_ascii=False, indent=2)

                    _log_kg_metrics(
                        tag="SLOW",
                        kg_payload=kg_payload,
                        findings_json=findings_json,
                        req_id=st.session_state.get("req_id"),
                    )

                    planner_meta = {
                        **(planner_meta or {}),
                        "use_thesis_replacement": bool(use_thesis_replacement),
                        "plan_courses": len(set(plan_course_ids)),
                        "kg_nodes_included": int((kg_payload.get("meta") or {}).get("node_count", 0)),
                        "kg_edges_included": int((kg_payload.get("meta") or {}).get("edge_count", 0)),
                        "kg_truncated_nodes": bool((kg_payload.get("meta") or {}).get("truncated_nodes", False)),
                        "kg_truncated_edges": bool((kg_payload.get("meta") or {}).get("truncated_edges", False)),
                    }
                    st.session_state.last_plan_meta = planner_meta
                    st.session_state.last_kg_payload = kg_payload
                    st.session_state.last_rag_chunks = []  # planning currently doesn't use retrieval evidence

                except Exception as exc:
                    logger.warning("Failed to build KG_FINDINGS for planning: %s", exc)

            synth_template = load_slow_planning_synth_prompt()
            evidence_block = ""

            prompt = synth_template.format(
                evidence=evidence_block,
                question=q,
                entities=json.dumps(entities_planning, ensure_ascii=False),
                intent=decision_dict.get("intent", "course_planning"),
                route=decision_dict.get("route", "SLOW"),
                confidence=str(decision_dict.get("confidence", "")),
                signals=", ".join(decision_dict.get("signals", []) or []),
                kg_findings=findings_json,
                curriculum_plan=curriculum_plan_str,
            )

            logger.debug(
                "Planning synth prompt breakdown | template=%d question=%d plan=%d kg_findings=%d TOTAL=%d",
                _safe_len(synth_template),
                _safe_len(q),
                _safe_len(curriculum_plan_str),
                _safe_len(findings_json),
                _safe_len(prompt),
            )

            # End timing right before returning the final answer to UI state.
            if not st.session_state.get("req_logged", True) and st.session_state.get("req_start_time") is not None:
                elapsed = time.perf_counter() - float(st.session_state.req_start_time)
                logger.info("Total request runtime: %.3f seconds", elapsed)
                st.session_state.req_logged = True

    if "last_question" in st.session_state:
        st.markdown("### Question")
        st.markdown(st.session_state.last_question)

    if st.session_state.last_plan is not None:
        st.subheader("Answer")
        render_plan(st.session_state.last_plan)

    if st.session_state.last_answer:
        st.subheader("Answer")
        st.write(st.session_state.last_answer)

    if not st.session_state.planning_active:
        render_evidence_panels(
            rag_chunks=st.session_state.get("last_rag_chunks"),
            kg_payload=st.session_state.get("last_kg_payload"),
        )

if __name__ == "__main__":
    main()
