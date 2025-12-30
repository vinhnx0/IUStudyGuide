# === stage3_ragkg/app/guardrails.py ===
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from app.logging_utils import get_logger, log_call

logger = get_logger(__name__)

# Email OR phone (international) with safe character classes (avoid ranges like \d-\s)
EMAIL_RE = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
PHONE_RE = r"(?:\+?\d{1,3})?(?:[\s\-\.]?\d){7,}"

PII_RE = re.compile(rf"({EMAIL_RE}|{PHONE_RE})")

INJECTION_PATTERNS = [
    r"(?i)\bignore\s+previous\s+instructions\b",
    r"(?i)\bdisregard\s+all\s+rules\b",
    r"(?i)\bpretend\s+to\s+be\b",
    r"(?i)\bchange\s+the\s+system\s+prompt\b",
    r"(?i)\boverride\b",
]


def strip_prompt_injection(text: str) -> str:
    """
    Remove known prompt-injection phrases from text.
    """
    cleaned = text
    for pat in INJECTION_PATTERNS:
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            logger.debug("strip_prompt_injection: removing pattern %r from text", pat)
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


@dataclass
class InputGuard:
    # FIX: use Tuple[str, ...] (ellipsis), not Tuple[str, .]
    allow_domains: Tuple[str, ...] = (
        "course",
        "curriculum",
        "subject",
        "prerequisite",
        "semester",
        "plan",
        "schedule",
        "đồ án",
        "luận văn",
        "học phần",
        "tín chỉ",
        "tiên quyết",
        "lộ trình",
    )

    @log_call(level=logging.DEBUG, include_result=False)
    def sanitize(self, text: str) -> Tuple[str, bool]:
        """
        Legacy API.
        Returns (clean_text, ok):
          - Removes prompt-injection phrases.
          - Redacts PII.
          - Domain allowlist light check (just a signal).
        """
        logger.debug("InputGuard.sanitize: original text length=%d", len(text))
        t = strip_prompt_injection(text)
        before_pii_len = len(t)
        t = PII_RE.sub("[REDACTED]", t)
        if len(t) != before_pii_len:
            logger.debug("InputGuard.sanitize: PII redacted from text")
        domain_ok = any(word in t.lower() for word in self.allow_domains)
        if not domain_ok:
            logger.info(
                "InputGuard.sanitize: text appears outside allowed domain; sample=%r",
                t[:120],
            )
        return t, domain_ok

    # NEW: wrapper for UI consistency
    @log_call(level=logging.DEBUG, include_result=True)
    def check(self, text: str) -> Dict[str, Any]:
        """
        New API used by Streamlit UI:
          returns {"ok": True/False, "clean_text": str, "issues": [..]}
        We do not hard-block; we just signal issues.
        """
        clean, domain_ok = self.sanitize(text)
        issues = []
        if not domain_ok:
            issues.append("outside_domain")
        if issues:
            logger.info("InputGuard.check: issues=%s", issues)
        return {"ok": True, "clean_text": clean, "issues": issues}


@dataclass
class OutputGuard:
    require_citations: bool = False

    @log_call(level=logging.DEBUG, include_result=True)
    def validate(self, text: str) -> Tuple[bool, str]:
        """
        Legacy API.
        Enforce that output contains a line starting with "Citations:".
        """
        if not self.require_citations:
            logger.debug("OutputGuard.validate: citations not required; skipping check")
            return True, text
        lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
        has_cite = any(x.lower().startswith("citations:") for x in lines)
        if not has_cite:
            logger.warning("OutputGuard.validate: output missing required citations")
            return False, "Output missing required citations."
        return True, text

    # NEW: wrapper for UI consistency
    @log_call(level=logging.DEBUG, include_result=True)
    def check(self, text: str) -> Dict[str, Any]:
        ok, msg = self.validate(text)
        return {"ok": ok, "message": msg if not ok else "ok"}
