# RAG Observability Logger — Runtime tracing for RAG pipelines, on real OpenTelemetry spans

A drop-in tracer for a retrieval-augmented-generation pipeline. Wrap your existing
retrieval, rerank, context-assembly, and generation calls; get a per-request trace
showing where time went and, when something breaks, which stage broke it — instead
of a single "the answer was bad" with no way to tell whether retrieval or the LLM
was at fault.

This is deliberately the *observability* half of a RAG system's reliability story,
not the *evaluation* half: it answers "what actually happened on this request,
right now" (runtime, every request), which is a different question from "is
retrieval quality good against a labeled set" (offline, evaluation frameworks like
RAGAS/DeepEval). Most write-ups about RAG show the pipeline; production systems
also need to show what the pipeline is *doing*.

## Design decisions

**Real OpenTelemetry spans, not a bespoke logger that imitates OTel's naming.**
The alternative — a custom `Span`/`Trace` class that just borrows `gen_ai.*` field
names — is faster to build, but the output only works with a reader you also wrote.
Building on the actual `opentelemetry-sdk` means the JSONL this produces comes from
real `ReadableSpan` objects: swapping `JSONLFileSpanExporter` for `OTLPSpanExporter`
is a one-line change to point this at Jaeger, Grafana Tempo, or any OTel collector —
nothing about the `traced_stage(...)` calls in your pipeline code would change.
Both major open-source LLM tracing tools (Arize Phoenix, via OpenInference; TruLens)
are themselves built on OTel, so this is the same foundation they use, not a
toy imitation of it.

**Field names follow OpenTelemetry's GenAI semantic conventions** (`gen_ai.*`) —
`gen_ai.operation.name`, `gen_ai.retrieval.query.text`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens`/`output_tokens`. These conventions are still pre-1.0 as
of 2026, but they're the direction the ecosystem is converging on, and using the
real names (rather than inventing your own) is what makes a trace file legible to
anyone who already knows this vocabulary.

**Retrieved documents are recorded as span *events*, one per document — not a
single JSON-blob attribute.** OTel span attributes are flat by design (a string,
number, bool, or a homogeneous array of one of those) — there's no way to attach a
nested object to a single attribute. The GenAI semantic conventions require
retrieval documents to be recorded in *structured* form on events specifically
(spans only get a JSON-string fallback). One event per document keeps each
document's `id`/`score`/`rank` independently queryable by anything reading the
trace, rather than a blob you can only grep.

**Pydantic validates the boundary, not the wire format.** `RetrievedDocument` is a
Pydantic model used *before* data gets flattened into OTel events — it's the type
safety on your side of the call, catching a malformed document shape early. Once
validated, it's flattened into plain OTel attributes, because that's the only shape
a span actually accepts. On the read side (`report.py`), `SpanRecord`/`TraceRecord`
Pydantic models parse the JSONL back into typed objects for rendering.

**Four fixed stages — retrieval → rerank → context assembly → generation — each
its own child span under one root span per request.** This mirrors how Langfuse
and Phoenix structure a trace (a top-level request containing per-stage
observations), so a trace here reads the same way traces in those tools do.
Nesting is automatic: OTel propagates trace/span IDs through the context manager
stack, so `record_retrieved_documents` never has to manually thread an ID through.

**Exceptions are recorded on the span, then re-raised — never swallowed.** A
failure in any stage shows up in the trace with `status=ERROR` and the exception
attached (via `span.record_exception`), and still propagates normally to your own
error handling. Silently swallowing the error to "keep the trace clean" would hide
exactly the failure you built this to catch.

**No server, no OTel Collector, no FastAPI.** The scope here is a tracer + a file
exporter + a report renderer — everything runs as a plain Python script. A live
dashboard over HTTP is a legitimate next step, not a requirement to prove the
tracing itself works.

## What gets captured, per stage

| Stage | Attributes | Events |
|---|---|---|
| retrieval | `gen_ai.data_source.id`, `retrieval.document_count`, `retrieval.top_score` | one `gen_ai.retrieval.documents` event per doc (`id`, `score`, `rank`) |
| rerank | `rerank.input_count`, `rerank.output_count` | — |
| context_assembly | `context.chunk_count`, `context.token_budget`, `context.total_tokens_used` | — |
| generation | `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | — |

Every span also carries `gen_ai.operation.name`, start/end time, duration, and a
status (`OK` or `ERROR`, with the exception attached on failure).

## Usage

```python
from tracer import configure, traced_request, traced_stage, record_retrieved_documents
from models import StageType, RetrievedDocument

configure("traces.jsonl")  # call once, at startup

with traced_request("rag_request"):
    with traced_stage(StageType.RETRIEVAL, "es_hybrid_search", **{"gen_ai.data_source.id": "my-index"}) as span:
        docs = my_retriever(query)  # -> list[RetrievedDocument]
        record_retrieved_documents(span, docs)

    with traced_stage(StageType.RERANK, "cross_encoder_rerank") as span:
        docs = my_reranker(docs)

    with traced_stage(StageType.CONTEXT_ASSEMBLY, "assemble_context") as span:
        context = my_context_builder(docs)

    with traced_stage(StageType.GENERATION, "llm_generate", **{"gen_ai.request.model": "gpt-4o-mini"}) as span:
        answer = my_llm_call(query, context)
```

## Try it in 30 seconds

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/demo_pipeline.py
python report.py examples/example_trace.jsonl --html examples/example_report.html
```

Example output (one clean run, one deliberately failed generation call — so you can
see what a broken stage looks like in the trace, not just the happy path):

```
Trace 6513270e269e  total=100.1ms
  es_hybrid_search             25.2ms   5 docs retrieved, top_score=0.775
  cross_encoder_rerank         12.6ms   5 -> 5 docs reranked
  assemble_context              6.3ms   5 chunks assembled, 60/200 tokens used
  llm_generate                 55.1ms   model=gpt-4o-mini, in=68 out=17 tokens

Trace 95e60af593bd  total=46.5ms
  es_hybrid_search             25.1ms   1 docs retrieved, top_score=0.57
  cross_encoder_rerank         12.6ms   1 -> 1 docs reranked
  assemble_context              6.3ms   1 chunks assembled, 13/200 tokens used
  llm_generate                  1.4ms   model=gpt-4o-mini, in=? out=? tokens  [ERROR]
```

## Tests

```bash
pip install pytest
pytest -v
```

## Requirements

- Python 3.10+
- `opentelemetry-api`, `opentelemetry-sdk`, `pydantic` (see `requirements.txt`)

## Natural next step (not built here)

Swap `JSONLFileSpanExporter` for `OTLPSpanExporter` to ship traces to a real
backend (Jaeger, Tempo, an OTel Collector) instead of a local file — the
instrumentation code above doesn't change at all, only `configure()`'s exporter.
