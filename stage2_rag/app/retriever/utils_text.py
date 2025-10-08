from __future__ import annotations

import re
import unicodedata


def normalize_vietnamese_text(s: str) -> str:
    s = s.replace("\u00A0", " ")  # nbsp
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    return s
