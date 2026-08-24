"""Answer generation: build RAG prompt and call the LLM."""

import os
from google import genai
from dotenv import load_dotenv

from . import config
from .vector_store import retrieve, get_collection
from .embeddings import create_embedding

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)


SYSTEM_PROMPT = """
You are a legal contract document assistant.

Answer the user's question ONLY using the
provided amendment documents.

Rules:

1. Do not use outside knowledge.
2. Do not invent contract terms.
3. Do not make assumptions.
4. If the answer cannot be found in the
   provided context, say:

"I could not find this information in the
provided amendment documents."

5. Give a concise answer.
6. Mention the relevant source and page.
"""


def _build_context(results):
    """Format retrieved chunks into a context string for the prompt."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"SOURCE: {metadata['source']}\n"
            f"PAGE: {metadata['page']}\n\n"
            f"{document}"
        )

    return "\n\n---\n\n".join(context_parts)


def _extract_sources(results):
    """Pull source metadata and chunk text from retrieval results for the response schema."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    rerank_scores = results.get("rerank_scores", [None] * len(distances))

    sources = []
    for document, metadata, distance, rerank_score in zip(documents, metadatas, distances, rerank_scores):
        source = {
            "source": metadata["source"],
            "page": metadata["page"],
            "heading": metadata.get("heading", ""),
            "distance": round(distance, 4),
            "text": document,
        }
        if rerank_score is not None:
            source["rerank_score"] = round(rerank_score, 4)
        sources.append(source)

    return sources


def generate_answer(question, results):
    """Generate an answer using retrieved context and the Gemini model.

    Returns plain text answer (backwards-compatible with the old interface).
    """

    context = _build_context(results)

    prompt = f"""{SYSTEM_PROMPT}

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=prompt
    )

    return response.text


class GeminiGenerator:
    """Generator class that wraps retrieval + reranking + generation.

    Pipeline: Chroma (RETRIEVAL_K) → Cross-Encoder (FINAL_K) → Gemini → answer
    """

    def __init__(self, model=None):
        self.model = model or config.GENERATION_MODEL

    def generate(self, query, where=None,
                 retrieval_k=config.RETRIEVAL_K,
                 mmr_k=config.MMR_K,
                 mmr_lambda=config.MMR_LAMBDA,
                 final_k=config.FINAL_K,
                 search_mode=config.SEARCH_MODE):
        """Retrieve, rerank, and generate an answer.

        Pipeline: Retrieval (retrieval_k) → MMR (mmr_k diverse) → Cross-Encoder (final_k) → LLM

        Args:
            query: The user's question.
            where: Optional ChromaDB metadata filter dict.
            retrieval_k: Number of initial candidates (Stage 1: RRF / search).
            mmr_k: Number of diverse chunks after MMR (Stage 2: diversity).
            mmr_lambda: MMR relevance-vs-diversity tradeoff (0.0–1.0).
            final_k: Number of chunks after cross-encoder reranking (Stage 3: accuracy).
            search_mode: "hybrid" (RRF fusion), "vector" (semantic only), or "bm25" (keyword only).

        Returns:
            Dict matching RESPONSE_SCHEMA + raw chunk lists for UI display.
        """
        from .reranker import rerank  # lazy import to avoid loading model at startup if unused
        from .mmr import mmr_rerank

        # Stage 1 — Retrieval (depends on search_mode)
        raw_vector_chunks = []
        raw_bm25_chunks = []

        if search_mode == "hybrid":
            from .hybrid_search import hybrid_retrieve
            results = hybrid_retrieve(query, top_k=retrieval_k, where=where)
            retrieval_label = "Hybrid (Vector + BM25 → RRF)"
            # Extract raw results attached by hybrid_retrieve
            raw_vector_chunks = self._format_raw_chunks(results.get("vector_results", {}), "vector")
            raw_bm25_chunks = self._format_raw_chunks(results.get("bm25_results", {}), "bm25")
        elif search_mode == "bm25":
            from .bm25_search import bm25_retrieve
            results = bm25_retrieve(query, top_k=retrieval_k)
            retrieval_label = "BM25 keyword search"
            raw_bm25_chunks = self._format_raw_chunks(results, "bm25")
        else:  # "vector"
            collection = get_collection(config.COLLECTION_NAME)
            results = retrieve(collection, query, retrieval_k, where=where)
            retrieval_label = "Vector (semantic) search"
            raw_vector_chunks = self._format_raw_chunks(results, "vector")

        # Check if we got any results
        if not results["documents"] or not results["documents"][0]:
            return {
                "answer": "I could not find relevant information in the provided documents.",
                "reasoning": "No matching chunks were retrieved from the vector store.",
                "sources": [],
                "vector_chunks": [],
                "bm25_chunks": [],
                "mmr_chunks": [],
                "confidence": "low",
                "out_of_scope": False,
            }

        # Stage 2 — MMR diversity selection (reduce retrieval_k → mmr_k diverse)
        mmr_results = mmr_rerank(query, results, mmr_k=mmr_k, mmr_lambda=mmr_lambda)
        raw_mmr_chunks = self._format_mmr_chunks(mmr_results)

        # Stage 3 — Accurate cross-encoder reranking (reduce mmr_k → final_k)
        reranked = rerank(query, mmr_results, final_k=final_k)

        # Carry forward RRF scores if present (from hybrid search)
        if "rrf_scores" in mmr_results:
            rrf_map = {}
            for doc, score in zip(mmr_results["documents"][0], mmr_results["rrf_scores"]):
                rrf_map[doc] = score
            reranked["rrf_scores"] = [
                rrf_map.get(doc, None) for doc in reranked["documents"][0]
            ]

        # Carry forward retrieval origins (vector / bm25 / both)
        if "retrieval_origins" in mmr_results:
            origin_map = {}
            for doc, origin in zip(mmr_results["documents"][0], mmr_results["retrieval_origins"]):
                origin_map[doc] = origin
            reranked["retrieval_origins"] = [
                origin_map.get(doc, search_mode) for doc in reranked["documents"][0]
            ]

        # Carry forward MMR scores
        if "mmr_scores" in mmr_results:
            mmr_map = {}
            for doc, score in zip(mmr_results["documents"][0], mmr_results["mmr_scores"]):
                mmr_map[doc] = score
            reranked["mmr_scores"] = [
                mmr_map.get(doc, None) for doc in reranked["documents"][0]
            ]

        # Stage 4 — LLM generation with the best chunks
        context = _build_context(reranked)
        sources = _extract_sources(reranked)

        # Attach all scores and retrieval origins to sources
        rrf_scores = reranked.get("rrf_scores", [])
        retrieval_origins = reranked.get("retrieval_origins", [])
        mmr_scores = reranked.get("mmr_scores", [])
        for i, source in enumerate(sources):
            if i < len(rrf_scores) and rrf_scores[i] is not None:
                source["rrf_score"] = round(rrf_scores[i], 6)
            if i < len(mmr_scores) and mmr_scores[i] is not None:
                source["mmr_score"] = round(mmr_scores[i], 6)
            if i < len(retrieval_origins):
                source["retrieved_by"] = retrieval_origins[i]
            else:
                source["retrieved_by"] = search_mode

        prompt = f"""{SYSTEM_PROMPT}

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}
"""

        response = client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        n_rrf = len(results.get("documents", [[]])[0])
        n_mmr = len(mmr_results.get("documents", [[]])[0])

        return {
            "answer": response.text,
            "reasoning": (
                f"{retrieval_label}: {n_rrf} candidates → "
                f"MMR top {n_mmr} diverse (λ={mmr_lambda}) → "
                f"reranked to top {len(sources)} → "
                f"generated answer using {self.model}."
            ),
            "sources": sources,
            "vector_chunks": raw_vector_chunks,
            "bm25_chunks": raw_bm25_chunks,
            "mmr_chunks": raw_mmr_chunks,
            "confidence": "high" if sources else "low",
            "out_of_scope": False,
        }

    @staticmethod
    def _format_raw_chunks(results, origin_label):
        """Convert a ChromaDB-style result dict into a flat list of chunk dicts for UI display."""
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            chunks.append({
                "rank": i + 1,
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "heading": meta.get("heading", ""),
                "score": round(float(dist), 4),
                "origin": origin_label,
            })
        return chunks

    @staticmethod
    def _format_mmr_chunks(results):
        """Convert MMR results into a flat list of chunk dicts for UI display."""
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        mmr_scores = results.get("mmr_scores", [])
        origins = results.get("retrieval_origins", [])

        chunks = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            chunks.append({
                "rank": i + 1,
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "heading": meta.get("heading", ""),
                "mmr_score": round(mmr_scores[i], 6) if i < len(mmr_scores) else None,
                "origin": origins[i] if i < len(origins) else "",
            })
        return chunks


def get_generator(backend=None):
    """Factory function to get a generator instance.

    Args:
        backend: Optional model name override. Defaults to config.GENERATION_MODEL.

    Returns:
        A GeminiGenerator instance with a .generate(query, where=) method.
    """
    return GeminiGenerator(model=backend)
