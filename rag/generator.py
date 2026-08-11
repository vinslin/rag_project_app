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

    sources = []
    for metadata, distance in zip(metadatas, distances):
        sources.append({
            "source": metadata["source"],
            "page": metadata["page"],
            "heading": metadata.get("heading", ""),
            "distance": round(distance, 4),
        })

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
    """Generator class that wraps retrieval + generation into a single .generate() call.

    Used by the pipeline module for end-to-end question answering.
    """

    def __init__(self, model=None):
        self.model = model or config.GENERATION_MODEL

    def generate(self, query, where=None, top_k=config.TOP_K):
        """Retrieve context and generate an answer.

        Args:
            query: The user's question.
            where: Optional ChromaDB metadata filter dict (e.g. {"document_type": "amendment"}).
            top_k: Number of chunks to retrieve.

        Returns:
            Dict matching RESPONSE_SCHEMA: {answer, reasoning, sources, confidence, out_of_scope}
        """
        collection = get_collection(config.COLLECTION_NAME)

        results = retrieve(collection, query, top_k, where=where)

        # Check if we got any results
        if not results["documents"] or not results["documents"][0]:
            return {
                "answer": "I could not find relevant information in the provided documents.",
                "reasoning": "No matching chunks were retrieved from the vector store.",
                "sources": [],
                "confidence": "low",
                "out_of_scope": False,
            }

        context = _build_context(results)
        sources = _extract_sources(results)

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
            "reasoning": f"Retrieved {len(sources)} chunks and generated answer using {self.model}.",
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
