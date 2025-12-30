from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from ..logging_utils import get_logger, log_call

SemesterType = Union[Literal["all"], List[int]]

logger = get_logger(__name__)


class Credits(BaseModel):
    total: int
    theory: int
    lab: int

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _check_totals(self):
        if self.total != self.theory + self.lab:
            raise ValueError(
                f"credits.total ({self.total}) must equal theory+lab ({self.theory}+{self.lab})"
            )
        return self


class Eligibility(BaseModel):
    min_credits: int = Field(..., ge=0, description="Minimum accumulated credits to enroll")
    note: Optional[str] = None

    model_config = {"extra": "allow"}  # tolerate harmless extras


class CourseNode(BaseModel):
    id: str
    name: Dict[str, str]
    credits: Credits
    area: List[str]
    language: str
    semester: SemesterType
    compulsory: Literal["Compulsory", "Elective"]
    source: Optional[str] = None
    version: Optional[str] = None

    # Optional eligibility metadata
    eligibility: Optional[Eligibility] = None

    # allow extra attributes you may add later
    model_config = {"extra": "allow"}

    @field_validator("id")
    @classmethod
    def _non_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("id must be non-empty")
        return v

    @field_validator("semester")
    @classmethod
    def _semester_ok(cls, v):
        if v == "all":
            return v
        if isinstance(v, list) and all(isinstance(x, int) and x > 0 for x in v):
            return v
        raise ValueError('semester must be "all" or a list of positive ints')

    @field_validator("language")
    @classmethod
    def _lang_non_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("language cannot be empty")
        return v


class Edge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    type: Literal["prerequisite"]
    source: Optional[str] = None
    version: Optional[str] = None

    model_config = {
        "populate_by_name": True,  # allow using from_ when constructing
        "extra": "allow",
    }


class Graph(BaseModel):
    nodes: List[CourseNode]
    edges: List[Edge]

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _validate_edges(self):
        ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.from_ not in ids:
                raise ValueError(f"edge.from '{e.from_}' does not exist in nodes")
            if e.to not in ids:
                raise ValueError(f"edge.to '{e.to}' does not exist in nodes")
            if e.type != "prerequisite":
                raise ValueError("edge.type must be 'prerequisite'")
        return self


@log_call(level=20, include_result=False)  # INFO
def load_graph(
    path: Union[str, Path],
) -> Tuple[Graph, Dict[str, CourseNode], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load and validate the curriculum graph.

    Returns:
      - Graph pydantic model
      - node_map: id -> CourseNode
      - prereq_in: to -> list of from (incoming prereqs)
      - prereq_out: from -> list of to (outgoing dependents)
    """
    path = Path(path)
    logger.info("load_graph: loading KG from %s", path)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("load_graph: failed to read or parse JSON from %s", path)
        raise

    try:
        graph = Graph(**data)
    except Exception:
        logger.exception("load_graph: Graph validation failed for %s", path)
        raise

    node_map: Dict[str, CourseNode] = {n.id: n for n in graph.nodes}
    prereq_in: Dict[str, List[str]] = {n.id: [] for n in graph.nodes}
    prereq_out: Dict[str, List[str]] = {n.id: [] for n in graph.nodes}

    for e in graph.edges:
        prereq_in[e.to].append(e.from_)
        prereq_out[e.from_].append(e.to)

    logger.info(
        "load_graph: loaded KG from %s nodes=%d edges=%d",
        path,
        len(node_map),
        len(graph.edges),
    )
    return graph, node_map, prereq_in, prereq_out
