"""Retrieval: TF-IDF cosine and BM25, standard library only.

Both retrievers share the same interface so they can be swapped in
the sweep without changing the runner:

    retriever = build_retriever(chunks, method="bm25")
    ranked = retriever.search(question, top_k=3)  # [(chunk_id, score), ...]

BM25 is implemented from the Robertson formula with the usual
defaults (k1=1.5, b=0.75). No embeddings and no vector database in
this version — that is a stated limitation, not a hidden gap.
"""
import math
from collections import Counter

from .text import tokens


def _idf(n_docs, doc_freq):
    return {t: math.log((n_docs - c + 0.5) / (c + 0.5) + 1) for t, c in doc_freq.items()}


class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        docs = [c["text"] for c in chunks]
        df = Counter()
        for doc in docs:
            df.update(set(tokens(doc)))
        self.idf = _idf(len(docs), df)
        self.vectors = []
        for doc in docs:
            toks = tokens(doc)
            tf = Counter(toks)
            denom = max(len(toks), 1)
            self.vectors.append({t: (c / denom) * self.idf[t] for t, c in tf.items()})

    def search(self, question, top_k=3):
        qtok = tokens(question)
        tf = Counter(qtok)
        denom = max(len(qtok), 1)
        qv = {t: (c / denom) * self.idf.get(t, 1.0) for t, c in tf.items()}
        qn = math.sqrt(sum(v * v for v in qv.values()))
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            dot = sum(qv.get(t, 0) * v for t, v in vec.items())
            vn = math.sqrt(sum(v * v for v in vec.values()))
            score = dot / (qn * vn) if qn and vn else 0.0
            scored.append((chunk["id"], score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]


class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        docs = [tokens(c["text"]) for c in chunks]
        df = Counter()
        for toks in docs:
            df.update(set(toks))
        self.idf = _idf(len(docs), df)
        self.doc_lens = [max(len(t), 1) for t in docs]
        self.avg_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.term_freqs = [Counter(t) for t in docs]

    def search(self, question, top_k=3):
        qtok = set(tokens(question))
        scored = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            for term in qtok:
                if term not in self.idf:
                    continue
                tf = self.term_freqs[i].get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avg_len)
                score += self.idf[term] * tf * (self.k1 + 1) / denom
            scored.append((chunk["id"], score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]


def build_retriever(chunks, method="tfidf"):
    if method == "tfidf":
        return TfidfRetriever(chunks)
    if method == "bm25":
        return BM25Retriever(chunks)
    raise ValueError(f"unknown retriever: {method!r}")
