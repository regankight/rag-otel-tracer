import json
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


def test_otel_sampler_env_var_does_not_drop_spans(tmp_path):
    # Fresh process: the global provider can only be set once per process.
    import os
    import subprocess
    import sys

    out = tmp_path / "t.jsonl"
    code = (
        "from tracer import configure, traced_request\n"
        f"configure({str(out)!r})\n"
        "with traced_request('req'): pass\n"
    )
    env = {**os.environ, "OTEL_TRACES_SAMPLER": "always_off"}
    root = Path(__file__).parent.parent
    subprocess.run([sys.executable, "-c", code], env=env, cwd=root, check=True)
    assert out.exists() and out.read_text().strip()
