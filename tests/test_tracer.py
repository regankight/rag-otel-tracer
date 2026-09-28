import json
import tempfile
from pathlib import Path

import pytest

from models import RetrievedDocument, StageType
from tracer import configure, record_retrieved_documents, traced_request, traced_stage

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
