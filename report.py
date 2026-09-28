"""Render a JSONL trace file (written by tracer.py) into a readable report.

    python report.py traces.jsonl
    python report.py traces.jsonl --html report.html
"""

from __future__ import annotations

import argparse
import html as html_module
import json
from collections import defaultdict
from pathlib import Path

from models import SpanRecord, TraceRecord


def load_traces(path: str | Path) -> list[TraceRecord]:
    spans_by_trace: dict[str, list[SpanRecord]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = SpanRecord.model_validate(json.loads(line))
            spans_by_trace[record.trace_id].append(record)

    traces = []
    for trace_id, spans in spans_by_trace.items():
        spans.sort(key=lambda s: s.start_time_ns)
        traces.append(TraceRecord(trace_id=trace_id, spans=spans))
    traces.sort(key=lambda t: t.spans[0].start_time_ns if t.spans else 0)
    return traces


def _stage_detail(span: SpanRecord) -> str:
    """One human-readable line summarizing what happened in a stage span."""
    attrs = span.attributes
    stage = attrs.get("gen_ai.operation.name")

    if stage == "retrieval":
        count = attrs.get("retrieval.document_count", "?")
        top = attrs.get("retrieval.top_score", "?")
        return f"{count} docs retrieved, top_score={top}"
    if stage == "rerank":
        return f"{attrs.get('rerank.input_count', '?')} -> {attrs.get('rerank.output_count', '?')} docs reranked"
    if stage == "context_assembly":
        chunks = attrs.get("context.chunk_count", "?")
        tokens = attrs.get("context.total_tokens_used", "?")
        budget = attrs.get("context.token_budget", "?")
        return f"{chunks} chunks assembled, {tokens}/{budget} tokens used"
    if stage == "generation":
        model = attrs.get("gen_ai.request.model", "?")
        in_tok = attrs.get("gen_ai.usage.input_tokens", "?")
        out_tok = attrs.get("gen_ai.usage.output_tokens", "?")
        return f"model={model}, in={in_tok} out={out_tok} tokens"
    return ""


def render_text(traces: list[TraceRecord]) -> str:
    lines = []
    for trace in traces:
        lines.append(f"Trace {trace.trace_id[:12]}  total={trace.duration_ms:.1f}ms")
        for span in trace.spans:
            if span.parent_span_id is None:
                continue  # root span itself carries no stage detail
            status_flag = "" if span.status == "OK" else f"  [{span.status}]"
            detail = _stage_detail(span)
            lines.append(f"  {span.name:<24} {span.duration_ms:>8.1f}ms   {detail}{status_flag}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_html(traces: list[TraceRecord]) -> str:
    rows = []
    for trace in traces:
        rows.append(f"<h2>Trace {html_module.escape(trace.trace_id[:12])} "
                     f"<span class='total'>total {trace.duration_ms:.1f}ms</span></h2>")
        rows.append("<table><tr><th>Stage</th><th>Duration</th><th>Detail</th><th>Status</th></tr>")
        for span in trace.spans:
            if span.parent_span_id is None:
                continue
            css = "" if span.status == "OK" else " class='error'"
            rows.append(
                f"<tr{css}><td>{html_module.escape(span.name)}</td>"
                f"<td>{span.duration_ms:.1f}ms</td>"
                f"<td>{html_module.escape(_stage_detail(span))}</td>"
                f"<td>{html_module.escape(span.status)}</td></tr>"
            )
        rows.append("</table>")

    body = "\n".join(rows)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RAG trace report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h2 {{ margin-top: 2rem; }}
  .total {{ font-weight: normal; color: #666; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
  th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; }}
  tr.error td {{ color: #b00020; }}
</style></head>
<body>
<h1>RAG trace report</h1>
{body}
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_file", help="Path to a JSONL trace file written by tracer.py")
    parser.add_argument("--html", metavar="OUT", help="Also write a static HTML report to this path")
    args = parser.parse_args()

    traces = load_traces(args.trace_file)
    print(render_text(traces), end="")

    if args.html:
        Path(args.html).write_text(render_html(traces), encoding="utf-8")
        print(f"Wrote HTML report to {args.html}")


if __name__ == "__main__":
    main()
