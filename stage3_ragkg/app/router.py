from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.logging_utils import get_logger, log_call
from app.llm import llm_generate_json

logger = get_logger(__name__)
import app.llm as _llm
logger.info("IMPORT CHECK router: app.llm file=%s", getattr(_llm, "__file__", "unknown"))
logger.info("IMPORT CHECK router: llm_generate_json module=%s", getattr(llm_generate_json, "__module__", "unknown"))

# Simple heuristic course ID pattern (e.g., IT001IU, CS201, MATH101, etc.)
COURSE_ID_RE = re.compile(
    r"\b([A-Z]{2,5}\d{2,4}[A-Z]{0,3})\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    """
    Routing decision contract for the rest of the app.
    """
    route: str                 # "FAST" | "SLOW"
    confidence: float
    entities: List[str]
    signals: List[str]
    intent: str = "generic_qna"


# ======================================================
# Heuristic helpers
# ======================================================

@log_call(level=logging.DEBUG, include_result=True)
def detect_entities(text: str, alias_normalizer: Optional[Any] = None) -> List[str]:
    if not text:
        return []

    found: List[str] = []
    for match in COURSE_ID_RE.findall(text):
        cid = str(match).upper().strip()
        if not cid:
            continue
        if alias_normalizer is not None:
            try:
                cid = alias_normalizer.normalize(cid)
            except Exception:
                logger.exception("alias_normalizer.normalize failed for %s", cid)
        if cid and cid not in found:
            found.append(cid)

    return found


@log_call(level=logging.DEBUG, include_result=True)
def infer_intent(question: str, entities: List[str]) -> str:
    """Lightweight intent classifier (heuristic).

    This intent label is used for:
      - logging and UI transparency
      - selecting an answer template for FAST answers
      - fallback when the optional LLM router does not provide an intent

    Intents:
      - course_planning
      - course_lookup
      - prerequisite_reasoning
      - generic_qna

    """
    q = (question or "").lower()
    has_entity = bool(entities)

    # Planning: multi-semester curriculum construction (SLOW).
    if any(
        kw in q
        for kw in [
            "kế hoạch học", "kế hoạch học", "lên kế hoạch", "lên kế hoạch",
            "lộ trình", "roadmap", "curriculum plan", "study plan",
            "học kỳ", "học kì", "semester",
            "từ hk", "đến hk", "hk1", "hk2", "hk3", "hk4", "hk5", "hk6", "hk7", "hk8",
        ]
    ):
        return "course_planning"

    # Prerequisite reasoning: explicit dependency/path to reach a target course.
    # Also handles legacy "eligibility-style" wording ("đăng ký được không", "đủ điều kiện", etc.).
    if any(
        kw in q
        for kw in [
            "tiên quyết", "prerequisite", "prereq", "điều kiện tiên quyết", "phụ thuộc",
            "được học", "có được học", "đăng ký được", "đăng kí được",
            "đủ điều kiện", "có đủ điều kiện", "đủ chuẩn", "được đăng ký",
            "eligible", "eligibility",
        ]
    ):
        return "prerequisite_reasoning"

    # Course lookup: details of a course (content/credits/what is).
    if has_entity and any(
        kw in q
        for kw in ["nội dung", "học gì", "what is", "syllabus", "tín chỉ", "credit", "môn này"]
    ):
        return "course_lookup"

    return "generic_qna"



@log_call(level=logging.DEBUG, include_result=True)
def route_heuristic(
    question: str,
    retrieval_scores: List[float],
    entities: List[str],
    fast_threshold: float = 0.70,
) -> RouteDecision:
    scores = [float(s) for s in (retrieval_scores or [])]
    max_score = max(scores) if scores else 0.0
    max_score_clamped = max(0.0, min(1.0, max_score))

    intent = infer_intent(question, entities)
    signals: List[str] = ["heuristic"]

    if not scores:
        route = "SLOW"
        signals.append("no_scores")
    else:
        if max_score >= fast_threshold:
            route = "FAST"
            signals.append("high_score")
        else:
            route = "SLOW"
            signals.append("low_score")

    # Only PLANNING is forced to SLOW.
    if intent in {"course_planning"}:
        route = "SLOW"
        signals.append("intent_forced_slow")

    return RouteDecision(
        route=route,
        confidence=max_score_clamped,
        entities=entities,
        signals=signals,
        intent=intent,
    )


# ======================================================
# Router prompt loader
# ======================================================

def _load_router_prompt_template() -> str:
    candidate_paths = [
        Path("app/prompts/router_prompt.txt"),
        Path("prompts/router_prompt.txt"),
        Path("router_prompt.txt"),
    ]
    for path in candidate_paths:
        if path.exists():
            return path.read_text(encoding="utf-8")

    raise FileNotFoundError("router_prompt.txt not found")


# ======================================================
# LLM ROUTER (Pydantic + step)
# ======================================================

@log_call(level=logging.INFO, include_result=True)
def route_llm(
    question: str,
    entities: List[str],
    confidence: float,
    cfg: Optional[Dict[str, Any]] = None,
    decomp_sub_questions: Optional[List[str]] = None,
    decomp_ops: Optional[List[str]] = None,
) -> Optional[RouteDecision]:
    """
    LLM router override using Structured JSON (router).
    """
    if not question or not isinstance(cfg, dict):
        return None

    try:
        template = _load_router_prompt_template()
        prompt = template.format(
            question=question,
            entities=json.dumps(entities or [], ensure_ascii=False),
            confidence=f"{confidence:.3f}",
            decomp_sub_questions=json.dumps(decomp_sub_questions or [], ensure_ascii=False),
            decomp_ops=json.dumps(decomp_ops or [], ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("route_llm: failed to build prompt: %s", exc)
        return None

    try:
        data = llm_generate_json(
            prompt=prompt,
            cfg=cfg,
            step="router",
            caller="router",
            max_tokens=256,
        )
    except Exception as exc:
        logger.warning("route_llm: LLM JSON failed: %s", exc)
        return None

    route = data["route"]
    intent = data.get("intent") or infer_intent(question, entities)
    signals = list(data.get("signals", [])) + ["llm-router"]
    # Backward-compatibility: older router prompts/models may still emit eligibility_checking.
    if str(intent).strip().lower() == "eligibility_checking":
        intent = "prerequisite_reasoning"
        signals.append("eligibility->prereq")


    return RouteDecision(
        route=route,
        confidence=confidence,
        entities=entities,
        signals=signals,
        intent=intent,
    )


# ======================================================
# MAIN ENTRY POINT
# ======================================================

@log_call(level=logging.INFO, include_result=True)
def decide_route(
    question: str,
    retrieval_scores: List[float],
    entities: List[str],
    use_llm_router: bool = False,
    fast_threshold: float = 0.70,
    cfg: Optional[Dict[str, Any]] = None,
    decomp_entities: Optional[List[str]] = None,
    kg_queries: Optional[List[Dict[str, Any]]] = None,
    decomp_sub_questions: Optional[List[str]] = None
) -> RouteDecision:
    """
    Decide FAST vs SLOW and pick a stable intent label.

    Policy:
      - Only intent == 'planning' forces SLOW.
      - Other intents stay FAST even if retrieval confidence is low (unless your heuristic chooses SLOW).
        (LLM router may override, but we still enforce 'planning only' for SLOW.)
    """
    question_clean = (question or "").strip()

    base = route_heuristic(
        question_clean,
        retrieval_scores=retrieval_scores,
        entities=entities,
        fast_threshold=fast_threshold,
    )

    # Merge entities from decomposition
    try:
        merged = set(base.entities or [])
        for e in (decomp_entities or []):
            if e:
                merged.add(str(e).strip())
        base.entities = sorted(merged)
    except Exception:
        logger.exception("Failed to merge entities")

    # Extract ops from KG queries
    ops: List[str] = []
    for item in (kg_queries or []):
        if isinstance(item, dict):
            op = str(item.get("op", "")).strip()
            if op:
                ops.append(op)

    signals = list(base.signals)

    has_plan_op = "plan_curriculum" in ops
    is_planning = has_plan_op or base.intent == "course_planning"

    if is_planning:
        base.route = "SLOW"
        base.intent = "course_planning"
        signals.append("planning_detected")
    else:
        base.route = "FAST"
        if "lookup_prereqs" in ops and base.intent == "generic_qna":
            base.intent = "prerequisite_reasoning"
            signals.append("kg-op:lookup_prereqs")
        elif "lookup_course" in ops and base.intent == "generic_qna":
            base.intent = "course_lookup"
            signals.append("kg-op:lookup_course")

    base.signals = signals

    logger.info(
        "ROUTE_DECISION_PRE_LLM route=%s intent=%s confidence=%.3f signals=%s entities=%s",
        base.route,
        base.intent,
        float(base.confidence),
        ",".join(base.signals or []),
        ",".join(base.entities or []),
    )


    if use_llm_router:
        llm_decision = route_llm(
            question_clean,
            base.entities,
            base.confidence,
            cfg=cfg,
            decomp_sub_questions=decomp_sub_questions,
            decomp_ops=ops,
        )
        if llm_decision:
            logger.info(
                "ROUTE_DECISION_FINAL route=%s intent=%s confidence=%.3f signals=%s entities=%s",
                llm_decision.route,
                llm_decision.intent,
                float(llm_decision.confidence),
                ",".join(llm_decision.signals or []),
                ",".join(llm_decision.entities or []),
            )

            return llm_decision

    logger.info(
        "ROUTE_DECISION_FINAL route=%s intent=%s confidence=%.3f signals=%s entities=%s",
        base.route,
        base.intent,
        float(base.confidence),
        ",".join(base.signals or []),
        ",".join(base.entities or []),
    )

    return base
