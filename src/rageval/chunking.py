"""Chunking strategies.

Three strategies, one interface, so the sweep in the runner can
compare them on the same corpus and the same questions:

- ``passage``: one chunk per passage (the original baseline).
- ``sentence``: one chunk per sentence — finer granularity, more chunks,
  more chances for a near-duplicate sentence to outrank the gold one.
- ``window``: overlapping 2-sentence windows, stride 1 — a middle
  ground that keeps a little local context.
"""
from .text import sentences


def chunk_corpus(corpus, strategy="passage"):
    """Return a list of chunks: {id, doc_id, text}."""
    if strategy not in {"passage", "sentence", "window"}:
        raise ValueError(f"unknown chunking strategy: {strategy!r}")
    chunks = []
    for doc in corpus:
        if strategy == "passage":
            chunks.append({"id": doc["id"], "doc_id": doc["id"], "text": doc["text"]})
        elif strategy == "sentence":
            for i, s in enumerate(sentences(doc["text"])):
                chunks.append({"id": f"{doc['id']}#s{i}", "doc_id": doc["id"], "text": s})
        else:  # window
            sents = sentences(doc["text"])
            if len(sents) == 1:
                chunks.append({"id": f"{doc['id']}#w0", "doc_id": doc["id"], "text": sents[0]})
            for i in range(len(sents) - 1):
                chunks.append(
                    {"id": f"{doc['id']}#w{i}", "doc_id": doc["id"],
                     "text": " ".join(sents[i:i + 2])}
                )
    return chunks
