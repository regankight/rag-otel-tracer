"""Drop-in tracing for a RAG pipeline, built on real OpenTelemetry spans.

Design decision: this wraps the actual `opentelemetry-sdk`, not a bespoke
logger that merely borrows OTel's attribute names. The payoff: the JSONL
this produces is generated from real `ReadableSpan` objects, so swapping
JSONLFileSpanExporter for OTLPSpanExporter is a one-line change to point
this at Jaeger, Grafana Tempo, or any other OTel collector -- nothing about
the instrumentation calls below would need to change.

Usage:

    from tracer import configure, record_query, record_retrieved_documents, traced_request, traced_stage
    from models import StageType

    configure("traces.jsonl")

    with traced_request("rag_request") as root:
        record_query(root, query)  # recorded only if content capture is on
        with traced_stage(StageType.RETRIEVAL, "es_hybrid_search") as span:
            docs = my_retriever(query)
            record_retrieved_documents(span, docs)
        ...
"""

from __future__ import annotations

import json
import os
import re
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Status, StatusCode

from models import RetrievedDocument, StageType

_TRACER_NAME = "rag_obs"
_configured = False

# The same switch the OpenTelemetry Python GenAI instrumentations use for prompt and
# response content. Off unless set to "true".
CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_capture_content = False
_redact: Callable[[str], str] | None = None


class JSONLFileSpanExporter(SpanExporter):
    """Appends each finished span to a JSONL file, one span per line.

    JSONL rather than a single JSON array so a trace file can be tailed or
    grown across multiple runs without re-writing the whole thing.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.path.open("a", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(_span_to_record(span)) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _span_to_record(span: ReadableSpan) -> dict:
    ctx = span.get_span_context()
    parent = span.parent
    start = span.start_time or 0
    end = span.end_time or start
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "name": span.name,
        "start_time_ns": start,
        "end_time_ns": end,
        "duration_ms": (end - start) / 1_000_000,
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
        "events": [
            {
                "name": event.name,
                "timestamp_ns": event.timestamp,
                "attributes": dict(event.attributes or {}),
            }
            for event in span.events
        ],
    }


def configure(
    output_path: str | Path = "traces.jsonl",
    *,
    capture_content: bool | None = None,
    redact: Callable[[str], str] | None = None,
) -> None:
    """Point the global tracer provider at a JSONL file. Call once, at startup.

    capture_content: record text content (the user's query) on spans. Off by
    default; None means read CAPTURE_CONTENT_ENV. The GenAI semantic conventions
    flag query text as possibly sensitive, and a trace backend usually has wider
    access than the application's own data.
    redact: applied to every captured text before it is recorded, e.g.
    basic_redactor or your own PII scrubber.

    Idempotent -- safe to call multiple times (e.g. once per test) without
    stacking up duplicate processors, since re-configuring a provider that's
    already set is a no-op in the OTel SDK. The content settings are updated
    on every call.
    """
    global _configured, _capture_content, _redact
    if capture_content is None:
        capture_content = os.environ.get(CAPTURE_CONTENT_ENV, "").strip().lower() == "true"
    _capture_content = capture_content
    _redact = redact
    if _configured:
        return
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JSONLFileSpanExporter(output_path)))
    trace.set_tracer_provider(provider)
    _configured = True


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DIGIT_RUN = re.compile(r"\+?\(?\d[\d ().-]*\d")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _phone_or_keep(match: re.Match) -> str:
    run = match.group(0)
    digits = sum(c.isdigit() for c in run)
    if 7 <= digits <= 15 and not _ISO_DATE.fullmatch(run):
        return "[PHONE]"
    return run


def basic_redactor(text: str) -> str:
    """Replace email addresses and phone-like numbers (7-15 digits, ISO dates kept)
    with placeholders.

    A starting point, not a PII detector: names and street addresses pass through,
    and long numeric IDs are treated as phone numbers. Use a dedicated tool
    (e.g. Microsoft Presidio) where that matters.
    """
    return _DIGIT_RUN.sub(_phone_or_keep, _EMAIL.sub("[EMAIL]", text))


def record_query(span: Span, text: str) -> None:
    """Record the retrieval query as gen_ai.retrieval.query.text, only if content
    capture is on, after redaction if configured."""
    if _capture_content:
        span.set_attribute("gen_ai.retrieval.query.text", _redact(text) if _redact else text)


_FILE_PATH = re.compile(r'File "(?:[^"]*[/\\])?([^"/\\]+)"')


def _record_exception(span: Span, exc: BaseException) -> None:
    """record_exception, with file names instead of absolute paths in the stack trace,
    so traces don't carry the machine's username or directory layout."""
    stacktrace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    span.record_exception(exc, attributes={"exception.stacktrace": _FILE_PATH.sub(r'File "\1"', stacktrace)})


def _tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def traced_request(name: str = "rag_request") -> Iterator[Span]:
    """The root span for one end-to-end RAG request; every stage nests under it."""
    with _tracer().start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        try:
            yield span
        except BaseException as exc:
            _record_exception(span, exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_status(Status(StatusCode.OK))


@contextmanager
def traced_stage(stage: StageType, name: str, **attributes: str | int | float | bool) -> Iterator[Span]:
    """One pipeline stage (retrieval/rerank/context_assembly/generation) as a child span.

    Records `gen_ai.operation.name`, times the block automatically, and on
    any exception records it on the span (status=ERROR + the exception
    itself) before re-raising -- the failure shows up in the trace instead
    of just crashing silently past your logging.
    """
    with _tracer().start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        span.set_attribute("gen_ai.operation.name", stage.value)
        for key, value in attributes.items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            _record_exception(span, exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def record_retrieved_documents(span: Span, documents: Sequence[RetrievedDocument]) -> None:
    """Attach a validated list of retrieved documents to a retrieval span.

    One OTel event per document (not a single JSON-blob attribute). The
    GenAI semantic conventions define gen_ai.retrieval.documents as a list
    of {id, score} objects, structured on events and preferably on spans,
    but span attributes can't hold nested objects, so on a span it would
    become a JSON string. One event per document, with gen_ai.document.*
    attribute names chosen by this project (not defined by the
    conventions), keeps each document's id/score/rank queryable on its own.
    """
    span.set_attribute("retrieval.document_count", len(documents))
    if documents:
        span.set_attribute("retrieval.top_score", documents[0].score)
    for doc in documents:
        span.add_event(
            "gen_ai.retrieval.documents",
            attributes={
                "gen_ai.document.id": doc.id,
                "gen_ai.document.score": doc.score,
                "gen_ai.document.rank": doc.rank,
            },
        )
