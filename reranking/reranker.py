from sentence_transformers import CrossEncoder

from core import config

cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)


def rerank(question: str, results: dict, final_k: int = config.FINAL_K) -> dict:
    """Rerank retrieval results with a cross-encoder and return the top final_k."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    pairs = [[question, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(documents, metadatas, distances, scores),
        key=lambda x: x[3],
        reverse=True,
    )[:final_k]

    return {
        "documents":    [[item[0] for item in ranked]],
        "metadatas":    [[item[1] for item in ranked]],
        "distances":    [[item[2] for item in ranked]],
        "rerank_scores": [float(item[3]) for item in ranked],
    }
