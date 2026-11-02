"""A toy retrieval -> rerank -> context_assembly -> generation pipeline,
instrumented with rag_obs, so the tracer has something real to trace.

Everything here is a fake/in-memory stand-in for a real pipeline (no live
vectorstore or LLM calls) -- the point is to show the *instrumentation*
pattern you'd wrap around your own retriever/reranker/LLM client, not to be
a real RAG system.

Run it directly to produce examples/example_trace.jsonl, then:

    python report.py examples/example_trace.jsonl
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import RetrievedDocument, StageType
from tracer import configure, record_query, record_retrieved_documents, traced_request, traced_stage

CORPUS = [
    {"id": "doc-1", "text": "Acme Cloud's autoscaler adds nodes when queue depth exceeds 50 for 2 minutes."},
    {"id": "doc-2", "text": "Acme Cloud bills compute per second, rounded up to the nearest 10 seconds."},
    {"id": "doc-3", "text": "Acme Cloud's object storage replicates writes across 3 availability zones by default."},
    {"id": "doc-4", "text": "Acme Cloud's CLI authenticates via a device-code flow, not static API keys."},
    {"id": "doc-5", "text": "Acme Cloud's autoscaler has a 5-minute cooldown between scale-down events."},
]


def fake_retrieve(query: str) -> list[RetrievedDocument]:
    """Stand-in for a hybrid (BM25 + vector) search call."""
    time.sleep(0.02)
    scored = [
        (doc, round(random.uniform(0.55, 0.97), 3))
        for doc in CORPUS
        if any(word.lower() in doc["text"].lower() for word in query.split())
    ] or [(doc, round(random.uniform(0.3, 0.6), 3)) for doc in CORPUS[:3]]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [
        RetrievedDocument(id=doc["id"], score=score, rank=i, text_preview=doc["text"][:80])
        for i, (doc, score) in enumerate(scored)
    ]


def fake_rerank(docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
    """Stand-in for a cross-encoder rerank pass -- nudges scores and re-sorts."""
    time.sleep(0.01)
    reranked = [doc.model_copy(update={"score": round(doc.score * random.uniform(0.9, 1.1), 3)}) for doc in docs]
    reranked.sort(key=lambda d: d.score, reverse=True)
    return [doc.model_copy(update={"rank": i}) for i, doc in enumerate(reranked)]


def fake_assemble_context(docs: list[RetrievedDocument], token_budget: int = 200) -> tuple[str, int]:
    """Stand-in for stitching top chunks into a prompt within a token budget."""
    time.sleep(0.005)
    parts: list[str] = []
    used = 0
    for doc in docs:
        cost = len(doc.text_preview.split())
        if used + cost > token_budget:
            break
        parts.append(doc.text_preview)
        used += cost
    return "\n".join(parts), used


def fake_generate(query: str, context: str) -> tuple[str, dict[str, int]]:
    """Stand-in for the LLM call."""
    time.sleep(0.05)
    answer = f"Based on the docs: {context.splitlines()[0] if context else 'no context available'}"
    usage = {"input": len(context.split()) + len(query.split()), "output": len(answer.split())}
    return answer, usage


def run_pipeline(query: str, fail_generation: bool = False) -> str:
    with traced_request("rag_request") as root:
        record_query(root, query)

        with traced_stage(StageType.RETRIEVAL, "es_hybrid_search", **{"gen_ai.data_source.id": "acme-docs-demo"}) as span:
            docs = fake_retrieve(query)
            record_retrieved_documents(span, docs)

        with traced_stage(StageType.RERANK, "cross_encoder_rerank", **{"rerank.input_count": len(docs)}) as span:
            docs = fake_rerank(docs)
            span.set_attribute("rerank.output_count", len(docs))

        with traced_stage(StageType.CONTEXT_ASSEMBLY, "assemble_context", **{"context.token_budget": 200}) as span:
            context, used_tokens = fake_assemble_context(docs)
            span.set_attribute("context.chunk_count", len(docs))
            span.set_attribute("context.total_tokens_used", used_tokens)

        with traced_stage(
            StageType.GENERATION,
            "llm_generate",
            **{"gen_ai.request.model": "gpt-4o-mini", "gen_ai.provider.name": "openai"},
        ) as span:
            if fail_generation:
                raise TimeoutError("generation call timed out after 30s")
            answer, usage = fake_generate(query, context)
            span.set_attribute("gen_ai.usage.input_tokens", usage["input"])
            span.set_attribute("gen_ai.usage.output_tokens", usage["output"])

    return answer


if __name__ == "__main__":
    trace_path = Path(__file__).parent / "example_trace.jsonl"
    trace_path.unlink(missing_ok=True)
    configure(trace_path)

    random.seed(7)  # reproducible example output for the committed trace file

    answer = run_pipeline("How does the autoscaler decide to add nodes?")
    print(f"Good run -> {answer}")

    try:
        run_pipeline("What does billing round to?", fail_generation=True)
    except TimeoutError as exc:
        print(f"Failed run -> caught and traced: {exc}")

    print(f"\nWrote trace to {trace_path}")
    print(f"Render it with: python report.py {trace_path}")
