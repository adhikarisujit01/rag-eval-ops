"""Reranking: a second, cheap pass over the top-N retrieved chunks.

No cross-encoder here — this is a transparent, deterministic reranker
that combines the retrieval score with direct keyword overlap between
the question and the chunk. It is meant to show *that* reranking can
change the order (and sometimes fix a near-duplicate mistake), not to
claim state-of-the-art reranking.
"""
from .text import tokens


def rerank(question, ranked, chunks_by_id, top_k=3, weight_retrieval=0.7):
    """ranked: [(chunk_id, score), ...] from a retriever.

    Returns the top_k chunk ids in reranked order, with scores.
    Scores are normalised per query (min-max over the candidate set)
    so the two signals are on a comparable scale.
    """
    if not ranked:
        return []
    qset = set(tokens(question))
    scores = [s for _, s in ranked]
    lo, hi = min(scores), max(scores)
    out = []
    for chunk_id, score in ranked:
        text = chunks_by_id[chunk_id]["text"]
        overlap = len(qset & set(tokens(text))) / max(len(qset), 1)
        norm = (score - lo) / (hi - lo) if hi > lo else 0.0
        combined = weight_retrieval * norm + (1 - weight_retrieval) * overlap
        out.append((chunk_id, round(combined, 4)))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:top_k]
