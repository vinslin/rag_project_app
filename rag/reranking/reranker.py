from sentence_transformers import CrossEncoder

from rag import config


cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)


def rerank(question, results, final_k=config.FINAL_K):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Build (question, chunk) pairs for the cross-encoder
    pairs = [
        [question, document]
        for document in documents
    ]

    # Score each pair — higher score = more relevant
    scores = cross_encoder.predict(pairs)

    # Sort by cross-encoder score (descending)
    ranked = sorted(
        zip(documents, metadatas, distances, scores),
        key=lambda x: x[3],
        reverse=True
    )

    # Keep only the top final_k
    ranked = ranked[:final_k]

    return {
        "documents": [[item[0] for item in ranked]],
        "metadatas": [[item[1] for item in ranked]],
        "distances": [[item[2] for item in ranked]],
        "rerank_scores": [float(item[3]) for item in ranked],
    }
