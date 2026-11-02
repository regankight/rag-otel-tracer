"""Trace a real RAG pipeline: real embeddings, real retrieval, real LLM calls.

The demo in demo_pipeline.py fakes every stage (random scores, sleep() timings).
This script wraps the tracer around working code instead:

- retrieval: the search module from rag-search-mcp-server (sentence-transformers
  all-MiniLM-L6-v2, cosine similarity over its bundled sample_docs/)
- context assembly: top chunks packed into a budget counted in the embedding
  model's own tokens
- generation: a local model through Ollama, with the token counts Ollama reports

There is no rerank stage, because this pipeline doesn't have one. The last request
asks for a model that isn't installed, so the error span comes from a real failure,
not a raised-on-purpose exception.

    python examples/trace_real_pipeline.py --search-repo ~/rag-search-mcp-server
    python report.py examples/real_trace.jsonl

Needs: sentence-transformers and ollama installed, and Ollama running with the
model pulled (default llama3.2). If Ollama isn't reachable, retrieval and context
assembly are still traced and generation shows up as an error span.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import RetrievedDocument, StageType  # noqa: E402
from tracer import configure, record_retrieved_documents, traced_request, traced_stage  # noqa: E402

QUESTIONS = [
    "How does hybrid search combine keyword and vector results?",
    "What should observability capture for a retrieval-augmented system?",
]
FAILING_QUESTION = "What is mean reciprocal rank?"
MISSING_MODEL = "model-that-is-not-installed"
TOP_K = 3
TOKEN_BUDGET = 400


def load_search(search_repo: Path):
    """Import search.py from a local rag-search-mcp-server checkout."""
    if not (search_repo / "search.py").exists():
        sys.exit(f"No search.py in {search_repo}. Pass --search-repo pointing at rag-search-mcp-server.")
    sys.path.insert(0, str(search_repo))
    import search

    return search


def chunk_key(search, index, i: int) -> str:
    return f"{index.sources[i]}#{search.format_chunk_id(i)}"


def retrieve(search, index, query: str) -> list[RetrievedDocument]:
    ranked = search.rank_chunks(query, index.embeddings, index.model, top_n=TOP_K)
    return [
        RetrievedDocument(
            id=chunk_key(search, index, i),
            score=round(score, 4),
            rank=rank,
            text_preview=index.chunks[i][:200],
        )
        for rank, (i, score) in enumerate(ranked)
    ]


def assemble_context(index, docs: list[RetrievedDocument], chunk_text: dict[str, str]) -> tuple[str, int, int]:
    """Pack whole chunks, best first, until the next one would exceed the token budget."""
    parts, used = [], 0
    for doc in docs:
        cost = len(index.model.tokenizer.encode(chunk_text[doc.id], add_special_tokens=False))
        if used + cost > TOKEN_BUDGET:
            break
        parts.append(chunk_text[doc.id])
        used += cost
    return "\n\n---\n\n".join(parts), len(parts), used


def generate(model_name: str, query: str, context: str):
    import ollama

    prompt = (
        "Using only the context below, answer the question in two sentences.\n"
        "If the answer isn't in the context, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n"
    )
    return ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])


def run_request(search, index, chunk_text: dict[str, str], query: str, model_name: str) -> str:
    with traced_request("rag_request") as root:
        root.set_attribute("gen_ai.retrieval.query.text", query)

        with traced_stage(
            StageType.RETRIEVAL,
            "embedding_search",
            **{"gen_ai.data_source.id": "rag-search-mcp-server/sample_docs", "retrieval.top_k": TOP_K},
        ) as span:
            docs = retrieve(search, index, query)
            record_retrieved_documents(span, docs)

        with traced_stage(
            StageType.CONTEXT_ASSEMBLY, "assemble_context", **{"context.token_budget": TOKEN_BUDGET}
        ) as span:
            context, n_chunks, used = assemble_context(index, docs, chunk_text)
            span.set_attribute("context.chunk_count", n_chunks)
            span.set_attribute("context.total_tokens_used", used)
            if n_chunks == 0:
                raise ValueError(f"no retrieved chunk fits the {TOKEN_BUDGET}-token context budget")

        with traced_stage(
            StageType.GENERATION,
            "ollama_chat",
            **{"gen_ai.request.model": model_name, "gen_ai.provider.name": "ollama"},
        ) as span:
            response = generate(model_name, query, context)
            span.set_attribute("gen_ai.response.model", response.model)
            span.set_attribute("gen_ai.usage.input_tokens", response.prompt_eval_count or 0)
            span.set_attribute("gen_ai.usage.output_tokens", response.eval_count or 0)
            return response.message.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--search-repo", type=Path, required=True, help="path to a rag-search-mcp-server checkout")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to generate with (default: llama3.2)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "real_trace.jsonl")
    args = parser.parse_args()

    search = load_search(args.search_repo.expanduser().resolve())
    args.out.unlink(missing_ok=True)
    configure(args.out)

    started = time.perf_counter()
    with traced_request("index_build") as span:
        index = search.build_index()
        span.set_attribute("index.chunk_count", len(index.chunks))
        span.set_attribute("index.source_count", len(set(index.sources)))
    chunk_text = {chunk_key(search, index, i): text for i, text in enumerate(index.chunks)}
    print(f"Index: {len(index.chunks)} chunks from {len(set(index.sources))} files "
          f"in {time.perf_counter() - started:.1f}s")

    for query in QUESTIONS:
        try:
            answer = run_request(search, index, chunk_text, query, args.model)
            print(f"\nQ: {query}\nA: {answer.strip()}")
        except Exception as exc:  # traced as an error span already; keep going
            print(f"\nQ: {query}\nRequest failed (traced; see the ERROR span): {type(exc).__name__}: {exc}")

    try:
        run_request(search, index, chunk_text, FAILING_QUESTION, MISSING_MODEL)
    except Exception as exc:
        print(f"\nQ: {FAILING_QUESTION}\nExpected failure (model not installed), traced: {type(exc).__name__}")

    print(f"\nWrote {args.out}\nRender it with: python report.py {args.out}")


if __name__ == "__main__":
    main()
