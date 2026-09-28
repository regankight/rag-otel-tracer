"""Typed shapes used at the boundaries of the tracer.

Pydantic validates data *before* it gets flattened into OpenTelemetry spans
(on the write side) and *after* it's read back out of the JSONL export (on
the read side, in report.py). OTel span attributes only accept flat
primitives (str/int/float/bool, or homogeneous arrays of those) -- there is
no way to attach a nested object directly to a span attribute. So structured
data like a list of retrieved documents is validated as a Pydantic model
first, then flattened into one OTel span *event* per document. See
tracer.record_retrieved_documents.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class StageType(str, Enum):
    """The four stages a RAG request is broken into.

    Matches the shape used by Langfuse/Phoenix-style tracing: each stage is
    its own span, nested under one root span per request, so a single trace
    shows where time (and failures) actually went.
    """

    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"


class RetrievedDocument(BaseModel):
    """One retrieved document, validated before being recorded on a span.

    Field names deliberately echo OpenTelemetry's GenAI semantic conventions
    (gen_ai.document.id / .score / .rank) so the same shape maps 1:1 onto
    the span events emitted for it.
    """

    id: str
    score: float
    rank: int
    text_preview: str = Field(default="", max_length=200)


class SpanRecord(BaseModel):
    """One exported span, as read back from the JSONL trace file."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time_ns: int
    end_time_ns: int
    duration_ms: float
    status: str
    attributes: dict
    events: list[dict]


class TraceRecord(BaseModel):
    """All spans belonging to one trace_id, sorted by start time."""

    trace_id: str
    spans: list[SpanRecord]

    @property
    def duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time_ns for s in self.spans)
        end = max(s.end_time_ns for s in self.spans)
        return (end - start) / 1_000_000

    @property
    def root(self) -> SpanRecord | None:
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None
