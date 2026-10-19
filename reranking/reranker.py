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

    pool = sorted(
        zip(documents, metadatas, distances, scores),
        key=lambda x: x[3],
        reverse=True,
    )
    top_k = list(pool[:final_k])

    # Amendment guarantee: if the pool contains amendment chunks but none made it
    # into top_k, force the highest-scoring amendment chunk in by replacing the
    # lowest-scoring chunk that is NOT an MSA/schedule base chunk (to preserve
    # original-clause context needed for version_comparison questions).
    top_k_types = {item[1].get("doc_type", "") for item in top_k}
    if "amendment" not in top_k_types:
        amendment_pool = [item for item in pool[final_k:]
                          if item[1].get("doc_type") == "amendment"]
        if amendment_pool:
            # Find the lowest-scoring non-base chunk to displace; fall back to last.
            base_types = {"msa", "schedule", "nda"}
            for i in range(len(top_k) - 1, -1, -1):
                if top_k[i][1].get("doc_type", "") not in base_types:
                    top_k[i] = amendment_pool[0]
                    break
            else:
                top_k[-1] = amendment_pool[0]

    ranked = top_k

    return {
        "documents":    [[item[0] for item in ranked]],
        "metadatas":    [[item[1] for item in ranked]],
        "distances":    [[item[2] for item in ranked]],
        "rerank_scores": [float(item[3]) for item in ranked],
    }
