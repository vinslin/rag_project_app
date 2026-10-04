"""Answer generation: build RAG prompt and call the LLM."""

from core import config
from retrieval.vector_store import retrieve, get_collection
from generation.prompts import SYSTEM_PROMPT
from week7.llm_client import get_client, groq_call_with_retry

client = get_client()


def _build_context(results: dict) -> str:
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    parts = []
    for document, metadata in zip(documents, metadatas):
        parts.append(
            f"SOURCE: {metadata['source']}\n"
            f"PAGE: {metadata['page']}\n\n"
            f"{document}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_sources(results: dict) -> list[dict]:
    documents    = results["documents"][0]
    metadatas    = results["metadatas"][0]
    distances    = results["distances"][0]
    rerank_scores = results.get("rerank_scores", [None] * len(distances))

    sources = []
    for document, metadata, distance, rerank_score in zip(documents, metadatas, distances, rerank_scores):
        source = {
            "source":   metadata["source"],
            "page":     metadata["page"],
            "heading":  metadata.get("heading", ""),
            "distance": round(distance, 4),
            "text":     document,
        }
        if rerank_score is not None:
            source["rerank_score"] = round(rerank_score, 4)
        sources.append(source)
    return sources


class GeminiGenerator:
    """Generator class that wraps retrieval + reranking + generation.

    Pipeline: Retrieval → MMR → Cross-Encoder → LLM → answer
    """

    def __init__(self, model: str | None = None):
        self.model = model or config.GENERATION_MODEL

    def generate(
        self,
        query: str,
        where: dict | None = None,
        retrieval_k: int = config.RETRIEVAL_K,
        mmr_k: int = config.MMR_K,
        mmr_lambda: float = config.MMR_LAMBDA,
        final_k: int = config.FINAL_K,
        search_mode: str = config.SEARCH_MODE,
        conversation_history: list | None = None,
    ) -> dict:
        from reranking.reranker import rerank
        from retrieval.mmr import mmr_rerank

        raw_vector_chunks: list = []
        raw_bm25_chunks: list = []

        if search_mode == "hybrid":
            from retrieval.hybrid_search import hybrid_retrieve
            results = hybrid_retrieve(query, top_k=retrieval_k, where=where)
            retrieval_label = "Hybrid (Vector + BM25 → RRF)"
            raw_vector_chunks = self._format_raw_chunks(results.get("vector_results", {}), "vector")
            raw_bm25_chunks   = self._format_raw_chunks(results.get("bm25_results",   {}), "bm25")
        elif search_mode == "bm25":
            from retrieval.bm25_search import bm25_retrieve
            results = bm25_retrieve(query, top_k=retrieval_k)
            retrieval_label = "BM25 keyword search"
            raw_bm25_chunks = self._format_raw_chunks(results, "bm25")
        else:
            collection = get_collection(config.COLLECTION_NAME)
            results = retrieve(collection, query, retrieval_k, where=where)
            retrieval_label = "Vector (semantic) search"
            raw_vector_chunks = self._format_raw_chunks(results, "vector")

        if not results["documents"] or not results["documents"][0]:
            return {
                "answer":       "I could not find relevant information in the provided documents.",
                "reasoning":    "No matching chunks were retrieved from the vector store.",
                "sources":      [],
                "vector_chunks": [],
                "bm25_chunks":  [],
                "mmr_chunks":   [],
                "confidence":   "low",
                "out_of_scope": False,
            }

        mmr_results    = mmr_rerank(query, results, mmr_k=mmr_k, mmr_lambda=mmr_lambda)
        raw_mmr_chunks = self._format_mmr_chunks(mmr_results)
        reranked       = rerank(query, mmr_results, final_k=final_k)

        # Carry forward RRF scores
        if "rrf_scores" in mmr_results:
            rrf_map = dict(zip(mmr_results["documents"][0], mmr_results["rrf_scores"]))
            reranked["rrf_scores"] = [rrf_map.get(d) for d in reranked["documents"][0]]

        # Carry forward retrieval origins
        if "retrieval_origins" in mmr_results:
            origin_map = dict(zip(mmr_results["documents"][0], mmr_results["retrieval_origins"]))
            reranked["retrieval_origins"] = [origin_map.get(d, search_mode) for d in reranked["documents"][0]]

        # Carry forward MMR scores
        if "mmr_scores" in mmr_results:
            mmr_map = dict(zip(mmr_results["documents"][0], mmr_results["mmr_scores"]))
            reranked["mmr_scores"] = [mmr_map.get(d) for d in reranked["documents"][0]]

        context = _build_context(reranked)
        sources = _extract_sources(reranked)

        rrf_scores        = reranked.get("rrf_scores", [])
        retrieval_origins = reranked.get("retrieval_origins", [])
        mmr_scores        = reranked.get("mmr_scores", [])
        for i, source in enumerate(sources):
            if i < len(rrf_scores) and rrf_scores[i] is not None:
                source["rrf_score"] = round(rrf_scores[i], 6)
            if i < len(mmr_scores) and mmr_scores[i] is not None:
                source["mmr_score"] = round(mmr_scores[i], 6)
            source["retrieved_by"] = retrieval_origins[i] if i < len(retrieval_origins) else search_mode

        prompt = f"""{SYSTEM_PROMPT}

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}
"""
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": prompt})
        response = groq_call_with_retry(client, model=self.model, messages=messages)

        n_rrf = len(results.get("documents", [[]])[0])
        n_mmr = len(mmr_results.get("documents", [[]])[0])

        return {
            "answer": response.choices[0].message.content,
            "reasoning": (
                f"{retrieval_label}: {n_rrf} candidates → "
                f"MMR top {n_mmr} diverse (λ={mmr_lambda}) → "
                f"reranked to top {len(sources)} → "
                f"generated answer using {self.model}."
            ),
            "sources":       sources,
            "vector_chunks": raw_vector_chunks,
            "bm25_chunks":   raw_bm25_chunks,
            "mmr_chunks":    raw_mmr_chunks,
            "confidence":    "high" if sources else "low",
            "out_of_scope":  False,
        }

    @staticmethod
    def _format_raw_chunks(results: dict, origin_label: str) -> list[dict]:
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas",  [[]])[0]
        dists = results.get("distances",  [[]])[0]
        return [
            {
                "rank":    i + 1,
                "text":    doc,
                "source":  meta.get("source", ""),
                "page":    meta.get("page", ""),
                "heading": meta.get("heading", ""),
                "score":   round(float(dist), 4),
                "origin":  origin_label,
            }
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists))
        ]

    @staticmethod
    def _format_mmr_chunks(results: dict) -> list[dict]:
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        mmr_scores = results.get("mmr_scores", [])
        origins   = results.get("retrieval_origins", [])
        return [
            {
                "rank":      i + 1,
                "text":      doc,
                "source":    meta.get("source", ""),
                "page":      meta.get("page", ""),
                "heading":   meta.get("heading", ""),
                "mmr_score": round(mmr_scores[i], 6) if i < len(mmr_scores) else None,
                "origin":    origins[i] if i < len(origins) else "",
            }
            for i, (doc, meta) in enumerate(zip(docs, metas))
        ]


def get_generator(backend: str | None = None) -> GeminiGenerator:
    """Factory — returns a GeminiGenerator instance."""
    return GeminiGenerator(model=backend)
