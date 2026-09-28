import json
from pathlib import Path

from report import load_traces, render_html, render_text

BASE_FIXTURE = [
    {
        "trace_id": "t1", "span_id": "root1", "parent_span_id": None,
        "name": "rag_request", "start_time_ns": 0, "end_time_ns": 100_000_000,
        "duration_ms": 100.0, "status": "OK", "attributes": {}, "events": [],
    },
    {
        "trace_id": "t1", "span_id": "s1", "parent_span_id": "root1",
        "name": "es_hybrid_search", "start_time_ns": 0, "end_time_ns": 20_000_000,
        "duration_ms": 20.0, "status": "OK",
        "attributes": {
            "gen_ai.operation.name": "retrieval",
            "retrieval.document_count": 3,
            "retrieval.top_score": 0.9,
        },
        "events": [],
    },
]


def _write_fixture(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "traces.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def test_load_traces_groups_by_trace_id(tmp_path):
    path = _write_fixture(tmp_path, BASE_FIXTURE)
    traces = load_traces(path)
    assert len(traces) == 1
    assert len(traces[0].spans) == 2
    assert traces[0].duration_ms == 100.0


def test_render_text_includes_stage_detail(tmp_path):
    path = _write_fixture(tmp_path, BASE_FIXTURE)
    text = render_text(load_traces(path))
    assert "es_hybrid_search" in text
    assert "3 docs retrieved, top_score=0.9" in text


def test_render_html_marks_errors(tmp_path):
    fixture = BASE_FIXTURE + [{
        "trace_id": "t1", "span_id": "s2", "parent_span_id": "root1",
        "name": "llm_generate", "start_time_ns": 20_000_000, "end_time_ns": 21_000_000,
        "duration_ms": 1.0, "status": "ERROR", "attributes": {}, "events": [],
    }]
    path = _write_fixture(tmp_path, fixture)
    out = render_html(load_traces(path))
    assert "class='error'" in out
    assert "llm_generate" in out
