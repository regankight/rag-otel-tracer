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
nested object to a single attribute. The GenAI semantic conventions define
`gen_ai.retrieval.documents` as a list of `{id, score}` objects: structured when
recorded on an event, and on a span preferably structured, with a JSON string as
the fallback where structured values aren't supported. Splitting the list into one
event per document, with `gen_ai.document.id`/`score`/`rank` attributes, is this
project's own design, not something the conventions require (they don't define
those per-document attribute names). It keeps each document's fields
independently queryable by anything reading the trace, rather than a JSON blob
you can only grep.

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
from tracer import configure, record_query, record_retrieved_documents, traced_request, traced_stage
from models import StageType, RetrievedDocument

configure("traces.jsonl")  # call once, at startup

with traced_request("rag_request") as root:
    record_query(root, query)  # only recorded if content capture is on (see below)
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

### Query text is off by default

The GenAI semantic conventions warn that `gen_ai.retrieval.query.text` may contain sensitive
information, and a tracing backend is usually readable by more people than the application's
own data. So `record_query` writes nothing unless content capture is switched on, either with
`configure(..., capture_content=True)` or with the environment variable the OpenTelemetry Python
GenAI instrumentations use:

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

Pass `redact=` to scrub text before it's recorded. `basic_redactor` replaces email addresses and
phone-like numbers with `[EMAIL]` and `[PHONE]`; it's a starting point, not a PII detector (names
and street addresses pass through, and long numeric IDs are treated as phone numbers). Timings, scores, token counts and document IDs are always recorded.
Exception messages are recorded as raised, so keep user content out of them. Stack traces name
files without their directories, so a committed trace doesn't reveal the machine's username or folder layout.

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

The demo's stages are stand-ins (random scores, `sleep()` timings), so these numbers
show the report format, not real performance. For real numbers, see below.

## Traced on a real pipeline

The demo above fakes every stage. [`examples/trace_real_pipeline.py`](examples/trace_real_pipeline.py)
wraps the same tracer around working code:

- **retrieval** from [rag-search-mcp-server](https://github.com/regankight/rag-search-mcp-server):
  `all-MiniLM-L6-v2` embeddings and cosine similarity over its five sample documents
- **context assembly** into a 400-token budget, counted with the embedding model's own tokenizer
- **generation** by `llama3.2` through Ollama, recording the token counts Ollama reports

There is no rerank stage, because that pipeline has none. The last request asks for a model
that isn't installed, so its error span comes from a real failure, not a deliberate `raise`.

```bash
pip install -r requirements.txt -r requirements-examples.txt
ollama pull llama3.2
python examples/trace_real_pipeline.py --search-repo ../rag-search-mcp-server
python report.py examples/real_trace.jsonl
```

One run on a MacBook Air, committed as [`examples/real_trace.jsonl`](examples/real_trace.jsonl).
The first trace is the one-off index build (loading the embedding model and embedding the chunks):

```
Trace 3c6c03de7d8c  total=6423.2ms

Trace c94be1b5dae3  total=4074.6ms
  embedding_search             17.7ms   3 docs retrieved, top_score=0.6751
  assemble_context              0.6ms   2 chunks assembled, 331/400 tokens used
  ollama_chat                4055.2ms   model=llama3.2, in=374 out=66 tokens

Trace 88624f637cc2  total=2510.2ms
  embedding_search            137.4ms   3 docs retrieved, top_score=0.6152
  assemble_context              0.8ms   2 chunks assembled, 327/400 tokens used
  ollama_chat                2370.6ms   model=llama3.2, in=385 out=76 tokens

Trace 3265dfaa4d55  total=71.6ms
  embedding_search             64.8ms   3 docs retrieved, top_score=0.3732
  assemble_context              0.7ms   2 chunks assembled, 331/400 tokens used
  ollama_chat                   4.8ms   model=model-that-is-not-installed, in=? out=? tokens  [ERROR]
```

What the trace shows:

- **Generation is nearly all of the latency:** 4,055 of 4,075 ms on the first request and
  2,371 of 2,510 ms on the second. Retrieval and context assembly never took more than 139 ms together.
- **The first LLM call was the slower one in both runs so far:** 4.1 s vs 2.4 s here, and 8.7 s vs
  2.1 s in an earlier run, for similar prompts (374 vs 385 input tokens). That fits Ollama loading
  the model on first use. The trace shows where the time went; it can't say why.
- **A missing model fails fast and visibly:** 4.8 ms, `status=ERROR`, with one exception event on
  the span, while the stages before it are recorded normally.
- **Retrieval varied from 18 ms to 137 ms** across the three requests. These are single runs
  on one laptop, not a benchmark.
- **No query text in the file:** content capture was off (the default), so the trace holds
  timings, scores, token counts and document IDs, but not what was asked.

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
