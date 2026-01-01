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
            - Nới lỏng một số **ràng buộc cố định** (ví dụ: giới hạn môn học được chọn, hoặc học kỳ bắt đầu)
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


def main() -> None:
    cfg = get_cfg()

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

        entities = detect_entities(question_clean, alias_normalizer=alias_normalizer)

        decomp = decompose_question(question_clean, cfg)
        decomp_entities = decomp.get("entities") or []
        decomp_sub_questions = decomp.get("sub_questions") or []
        kg_queries = decomp.get("kg_queries") or []
        num_sub_questions = len(decomp_sub_questions)

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
            num_sub_questions=num_sub_questions,
        )

        st.session_state.last_route_decision = asdict(decision)

        is_planning = decision.route == "SLOW" and decision.intent == "course_planning"
        st.session_state.planning_active = bool(is_planning)
        st.session_state.planning_question = question_clean
        st.session_state.planning_decision = asdict(decision)
        st.session_state.planning_entities = list(decision.entities or [])
        st.session_state.planning_decomp = decomp
        st.session_state.planning_retrieval_scores = retrieval_scores
        st.session_state.last_answer = False

        st.session_state.last_plan = None
        st.session_state.last_plan_meta = {}

        if not is_planning:
            # Build compact KG_FINDINGS: relevant nodes + edges (+ derived per-target fields).
            kg_payload = build_relevant_kg_findings(
                seed_course_ids=list(decision.entities or []),
                node_map=node_map,
                prereq_in=prereq_in,
                include_prereq_depth=2,
                include_target_findings=True,
            )

            logger.info(
                "KG_FINDINGS(FAST) meta=%s",
                kg_payload.get("meta") or {},
            )

            findings_json = json.dumps(kg_payload, ensure_ascii=False, indent=2)

            synth_template = load_answer_synth_prompt()

            prompt = synth_template.format(
                evidence=evidence_block,
                question=question_clean,
                entities=json.dumps(decision.entities or [], ensure_ascii=False),
                intent=decision.intent,
                route=decision.route,
                confidence=f"{decision.confidence:.3f}",
                signals=", ".join(decision.signals or []),
                kg_findings=findings_json,
                curriculum_plan="",
            )

            logger.info(
                "LLM prompt breakdown | template=%d question=%d evidence=%d kg=%d TOTAL=%d",
                _safe_len(synth_template),
                _safe_len(question_clean),
                _safe_len(evidence_block),
                _safe_len(findings_json),
                _safe_len(prompt),
            )

            answer = llm_generate_text(prompt, cfg)
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

    if st.session_state.last_route_decision is not None:
        st.subheader("Route Decision")
        st.json(st.session_state.last_route_decision)

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

            logger.info(
                "Planning synth prompt breakdown | template=%d question=%d plan=%d kg_findings=%d TOTAL=%d",
                _safe_len(synth_template),
                _safe_len(q),
                _safe_len(curriculum_plan_str),
                _safe_len(findings_json),
                _safe_len(prompt),
            )
            
            
            # answer = llm_generate_text(prompt, cfg)
            # answer = strip_think(answer)

            # out_res = output_guard.check(answer)
            # if not out_res.get("ok", True):
            #     st.error("Output rejected by guardrails.")
            #     st.session_state.last_answer = ""
            # else:
            #     st.session_state.last_answer = answer

            # End timing right before returning the final answer to UI state.
            if not st.session_state.get("req_logged", True) and st.session_state.get("req_start_time") is not None:
                elapsed = time.perf_counter() - float(st.session_state.req_start_time)
                logger.info("Total request runtime: %.3f seconds", elapsed)
                st.session_state.req_logged = True

    if st.session_state.last_plan is not None:
        render_plan(st.session_state.last_plan)

    if st.session_state.last_plan_meta:
        st.subheader("Planner meta")
        st.json(st.session_state.last_plan_meta)

    if st.session_state.last_answer:
        st.subheader("Answer")
        st.write(st.session_state.last_answer)


if __name__ == "__main__":
    main()
