import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from models import RetrievedDocument, StageType
from tracer import (
    CAPTURE_CONTENT_ENV,
    basic_redactor,
    configure,
    record_query,
    record_retrieved_documents,
    traced_request,
    traced_stage,
)

TRACE_FILE = Path(tempfile.mkdtemp()) / "test_traces.jsonl"
configure(TRACE_FILE)  # OTel's global provider can only be set once per process,
                        # so all tests in this file share one trace file and
                        # each test filters to the span names it produced.


def _read_spans() -> list[dict]:
    with TRACE_FILE.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _span_named(name: str) -> dict:
    matches = [s for s in _read_spans() if s["name"] == name]
    assert len(matches) == 1, f"expected exactly one span named {name!r}, found {len(matches)}"
    return matches[0]


def test_stage_records_duration_and_ok_status():
    with traced_request("req"):
        with traced_stage(StageType.RETRIEVAL, "unique_span_success"):
            pass

    span = _span_named("unique_span_success")
    assert span["status"] == "OK"
    assert span["duration_ms"] >= 0
    assert span["attributes"]["gen_ai.operation.name"] == "retrieval"


def test_exception_in_stage_sets_error_status_and_reraises():
    with pytest.raises(ValueError):
        with traced_request("req"):
            with traced_stage(StageType.GENERATION, "unique_span_failure"):
                raise ValueError("boom")

    span = _span_named("unique_span_failure")
    assert span["status"] == "ERROR"
    assert any(e["name"] == "exception" for e in span["events"])


def test_record_retrieved_documents_adds_one_event_per_doc():
    docs = [
        RetrievedDocument(id="a", score=0.9, rank=0),
        RetrievedDocument(id="b", score=0.5, rank=1),
    ]
    with traced_request("req"):
        with traced_stage(StageType.RETRIEVAL, "unique_span_docs") as span:
            record_retrieved_documents(span, docs)

    span = _span_named("unique_span_docs")
    assert span["attributes"]["retrieval.document_count"] == 2
    assert span["attributes"]["retrieval.top_score"] == 0.9
    assert len(span["events"]) == 2
    assert span["events"][0]["attributes"]["gen_ai.document.id"] == "a"
    assert span["events"][1]["attributes"]["gen_ai.document.id"] == "b"


def test_parent_child_share_trace_id():
    with traced_request("unique_req_parent_child") as root:
        with traced_stage(StageType.RETRIEVAL, "unique_span_child"):
            pass

    root_span = _span_named("unique_req_parent_child")
    child_span = _span_named("unique_span_child")
    assert child_span["trace_id"] == root_span["trace_id"]
    assert child_span["parent_span_id"] == root_span["span_id"]


def test_query_text_is_not_recorded_by_default(monkeypatch):
    monkeypatch.delenv(CAPTURE_CONTENT_ENV, raising=False)
    configure(TRACE_FILE)
    with traced_request("unique_req_content_off") as root:
        record_query(root, "what is my balance?")

    assert "gen_ai.retrieval.query.text" not in _span_named("unique_req_content_off")["attributes"]


def test_query_text_is_recorded_when_the_env_switch_is_on(monkeypatch):
    monkeypatch.setenv(CAPTURE_CONTENT_ENV, "true")
    configure(TRACE_FILE)
    with traced_request("unique_req_content_env") as root:
        record_query(root, "what is my balance?")

    assert _span_named("unique_req_content_env")["attributes"]["gen_ai.retrieval.query.text"] == "what is my balance?"
    configure(TRACE_FILE, capture_content=False)


def test_redactor_runs_before_the_query_is_recorded():
    configure(TRACE_FILE, capture_content=True, redact=basic_redactor)
    with traced_request("unique_req_content_redacted") as root:
        record_query(root, "email jane.doe@example.com or call +1 (555) 010-2030 about order 42")
    configure(TRACE_FILE, capture_content=False)

    recorded = _span_named("unique_req_content_redacted")["attributes"]["gen_ai.retrieval.query.text"]
    assert recorded == "email [EMAIL] or call [PHONE] about order 42"


def test_stack_traces_name_files_without_their_directories():
    with pytest.raises(ValueError):
        with traced_request("unique_req_stacktrace"):
            with traced_stage(StageType.GENERATION, "unique_span_stacktrace"):
                raise ValueError("boom")

    for name in ("unique_req_stacktrace", "unique_span_stacktrace"):
        events = [e for e in _span_named(name)["events"] if e["name"] == "exception"]
        assert len(events) == 1
        stacktrace = events[0]["attributes"]["exception.stacktrace"]
        assert 'File "test_tracer.py"' in stacktrace
        assert str(Path(__file__).parent) not in stacktrace


def _run_fresh(tmp_path, body: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run code in a new interpreter: the global tracer provider can only be set once
    per process, so these tests can't share this module's configure()."""
    out = tmp_path / "t.jsonl"
    code = f"from tracer import *\nfrom models import *\nOUT = {str(out)!r}\n" + body
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )


def test_otel_sampler_env_var_does_not_drop_spans(tmp_path):
    body = "configure(OUT)\nwith traced_request('req'): pass\n"
    result = _run_fresh(tmp_path, body, {"OTEL_TRACES_SAMPLER": "always_off"})
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "t.jsonl").read_text().strip()


def test_otel_limit_env_vars_do_not_drop_or_truncate(tmp_path):
    body = (
        "configure(OUT)\n"
        "docs = [RetrievedDocument(id=f'doc-number-{i}', score=0.5, rank=i) for i in range(200)]\n"
        "with traced_request('req'):\n"
        "    with traced_stage(StageType.RETRIEVAL, 'retrieve') as span:\n"
        "        record_retrieved_documents(span, docs)\n"
    )
    limits = {
        "OTEL_SPAN_EVENT_COUNT_LIMIT": "1",
        "OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT": "1",
        "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT": "1",
        "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT": "3",
        "OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT": "3",
    }
    result = _run_fresh(tmp_path, body, limits)
    assert result.returncode == 0, result.stderr
    spans = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    retrieve = next(s for s in spans if s["name"] == "retrieve")
    assert len(retrieve["events"]) == 200  # also more than OTel's default cap of 128
    assert retrieve["events"][199]["attributes"] == {
        "gen_ai.document.id": "doc-number-199",
        "gen_ai.document.score": 0.5,
        "gen_ai.document.rank": 199,
    }
    assert retrieve["attributes"]["gen_ai.operation.name"] == "retrieval"


def test_configure_fails_loudly_if_another_provider_is_already_set(tmp_path):
    body = (
        "from opentelemetry import trace\n"
        "from opentelemetry.sdk.trace import TracerProvider\n"
        "trace.set_tracer_provider(TracerProvider())\n"
        "configure(OUT)\n"
    )
    result = _run_fresh(tmp_path, body, {})
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
