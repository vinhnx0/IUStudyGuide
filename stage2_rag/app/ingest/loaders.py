from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader
import markdown


def load_pdf(path: Path) -> List[Tuple[str, Dict]]:
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out.append((text, {"page": i + 1}))
    return out


def load_html(path: Path) -> List[Tuple[str, Dict]]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    return [(text, {"url": None})]


def load_md(path: Path) -> List[Tuple[str, Dict]]:
    md_text = path.read_text(encoding="utf-8")
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    return [(text, {"page": None})]


def load_csv(path: Path) -> List[Tuple[str, Dict]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    # Render as text blocks
    blocks = []
    for r in rows:
        items = [f"{k}: {v}" for k, v in r.items()]
        blocks.append("\n".join(items))
    return [(b, {"page": None}) for b in blocks]


def dispatch_load(path: Path) -> List[Tuple[str, Dict]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".html", ".htm"):
        return load_html(path)
    if suffix in (".md", ".markdown"):
        return load_md(path)
    if suffix == ".csv":
        return load_csv(path)
    # default: read as plain text
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, {"page": None})]
