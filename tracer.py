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
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, Sampler
from opentelemetry.trace import Span, Status, StatusCode

from models import RetrievedDocument, StageType

_TRACER_NAME = "rag_obs"
_configured = False
_configured_path: Path | None = None

# The same switch the OpenTelemetry Python GenAI instrumentations use for prompt and
# response content. Off unless set to "true".
CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_capture_content = False
_redact: Callable[[str], str] | None = None

# Every field set, because SpanLimits() fills any field left as None from an OTEL_*_LIMIT
# environment variable. Events and attribute lengths are unlimited so every retrieved
# document and the full stack trace reach the file; counts keep the OTel default of 128.
DEFAULT_SPAN_LIMITS = SpanLimits(
    max_attributes=128,
    max_span_attributes=128,
    max_event_attributes=128,
    max_link_attributes=128,
    max_links=128,
    max_events=SpanLimits.UNSET,
    max_attribute_length=SpanLimits.UNSET,
    max_span_attribute_length=SpanLimits.UNSET,
)


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
    sampler: Sampler = ALWAYS_ON,
    span_limits: SpanLimits = DEFAULT_SPAN_LIMITS,
) -> None:
    """Point the global tracer provider at a JSONL file. Call once, at startup.

    capture_content: record text content (the user's query) on spans. Off by
    default; None means read CAPTURE_CONTENT_ENV. The GenAI semantic conventions
    flag query text as possibly sensitive, and a trace backend usually has wider
    access than the application's own data.
    redact: applied to every captured text before it is recorded, e.g.
    basic_redactor or your own PII scrubber.
    sampler: which requests get recorded. Defaults to ALWAYS_ON, set explicitly
    so the OTEL_TRACES_SAMPLER environment variable can't silently drop spans
    (TracerProvider() with no sampler reads it). Pass e.g.
    ParentBased(TraceIdRatioBased(0.1)) to sample on purpose.
    span_limits: caps on attributes/events per span. Defaults to DEFAULT_SPAN_LIMITS,
    set explicitly for the same reason (OTEL_*_LIMIT variables would otherwise apply).
    OTEL_SDK_DISABLED=true is still honoured: it is OpenTelemetry's deliberate off switch.

    Raises RuntimeError if another global tracer provider was already set (by your
    app, or by running under opentelemetry-instrument): OTel keeps the first one,
    so this tracer's spans would never reach the file.

    Idempotent -- safe to call multiple times (e.g. once per test) with the same
    output_path; the content settings are updated on every call. A later call with
    a different output_path raises ValueError, since the exporter can't be swapped
    once the global provider is set and spans would keep going to the first file.
    """
    global _configured, _configured_path, _capture_content, _redact
    path = Path(output_path).resolve()
    if _configured and path != _configured_path:
        raise ValueError(
            f"configure() already sent traces to {_configured_path}; it can't switch to "
            f"{path} in the same process."
        )
    if capture_content is None:
        capture_content = os.environ.get(CAPTURE_CONTENT_ENV, "").strip().lower() == "true"
    _capture_content = capture_content
    _redact = redact
    if _configured:
        return
    provider = TracerProvider(sampler=sampler, span_limits=span_limits)
    provider.add_span_processor(SimpleSpanProcessor(JSONLFileSpanExporter(path)))
    trace.set_tracer_provider(provider)
    if trace.get_tracer_provider() is not provider:
        raise RuntimeError(
            "A global OpenTelemetry tracer provider was already set, so configure() "
            "can't install its JSONL exporter and no trace file would be written."
        )
    _configured = True
    _configured_path = path


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DIGIT_RUN = re.compile(r"\+?\(?\d[\d ().-]*\d")
# Matched before _DIGIT_RUN so their digits aren't read as a phone number:
# ISO dates (optionally with a time), clock times, and IPv4 addresses.
_DATE_OR_TIME = r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?|\b\d{1,2}:\d{2}(?::\d{2})?\b"
_IPV4 = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
_NUMBERS = re.compile(rf"(?P<keep>{_DATE_OR_TIME})|(?P<ip>{_IPV4})|(?P<run>{_DIGIT_RUN.pattern})")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _redact_number(match: re.Match) -> str:
    if match.group("keep"):
        return match.group(0)
    if match.group("ip"):
        return "[IP]"
    run = match.group(0)
    digits = "".join(c for c in run if c.isdigit())
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return "[CARD]"
    if 7 <= len(digits) <= 15:
        return "[PHONE]"
    return run


def basic_redactor(text: str) -> str:
    """Replace email addresses, payment-card numbers (13-19 digits passing the Luhn
    check), IPv4 addresses and phone-like numbers (7-15 digits) with placeholders.
    Dates and clock times are kept.

    A starting point, not a PII detector: names and street addresses pass through,
    and long numeric IDs are treated as phone numbers. Use a dedicated tool
    (e.g. Microsoft Presidio) where that matters.
    """
    return _NUMBERS.sub(_redact_number, _EMAIL.sub("[EMAIL]", text))


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
        except BaseException as exc:  # incl. KeyboardInterrupt, like traced_request
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
        span.set_attribute("retrieval.top_score", max(doc.score for doc in documents))
    for doc in documents:
        span.add_event(
            "gen_ai.retrieval.documents",
            attributes={
                "gen_ai.document.id": doc.id,
                "gen_ai.document.score": doc.score,
                "gen_ai.document.rank": doc.rank,
            },
        )
