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
    """Pull source metadata from retrieval results for the response schema."""

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    rerank_scores = results.get("rerank_scores", [None] * len(distances))

    sources = []
    for metadata, distance, rerank_score in zip(metadatas, distances, rerank_scores):
        source = {
            "source": metadata["source"],
            "page": metadata["page"],
            "heading": metadata.get("heading", ""),
            "distance": round(distance, 4),
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
                 final_k=config.FINAL_K):
        """Retrieve, rerank, and generate an answer.

        Args:
            query: The user's question.
            where: Optional ChromaDB metadata filter dict.
            retrieval_k: Number of initial candidates from Chroma (Stage 1).
            final_k: Number of chunks kept after cross-encoder reranking (Stage 2).

        Returns:
            Dict matching RESPONSE_SCHEMA: {answer, reasoning, sources, confidence, out_of_scope}
        """
        from .reranker import rerank  # lazy import to avoid loading model at startup if unused

        collection = get_collection(config.COLLECTION_NAME)

        # Stage 1 — Fast bi-encoder retrieval (broad net)
        results = retrieve(collection, query, retrieval_k, where=where)

        # Check if we got any results
        if not results["documents"] or not results["documents"][0]:
            return {
                "answer": "I could not find relevant information in the provided documents.",
                "reasoning": "No matching chunks were retrieved from the vector store.",
                "sources": [],
                "confidence": "low",
                "out_of_scope": False,
            }

        # Stage 2 — Accurate cross-encoder reranking (precise scoring)
        reranked = rerank(query, results, final_k=final_k)

        # Stage 3 — LLM generation with the best chunks
        context = _build_context(reranked)
        sources = _extract_sources(reranked)

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

        return {
            "answer": response.text,
            "reasoning": (
                f"Retrieved {retrieval_k} candidates → "
                f"reranked to top {len(sources)} → "
                f"generated answer using {self.model}."
            ),
            "sources": sources,
            "confidence": "high" if sources else "low",
            "out_of_scope": False,
        }


def get_generator(backend=None):
    """Factory function to get a generator instance.

    Args:
        backend: Optional model name override. Defaults to config.GENERATION_MODEL.

    Returns:
        A GeminiGenerator instance with a .generate(query, where=) method.
    """
    return GeminiGenerator(model=backend)
