import json
from pathlib import Path

from rageval.chunking import chunk_corpus
from rageval.retrieval import build_retriever

DATA = Path(__file__).resolve().parents[1] / "data"


def _load():
    corpus = json.loads((DATA / "corpus.json").read_text())
    golden = json.loads((DATA / "golden.json").read_text())
    return corpus, golden


def test_retriever_returns_gold_in_top_k_for_easy_cases():
    corpus, golden = _load()
    for method in ("tfidf", "bm25"):
        chunks = chunk_corpus(corpus, "passage")
        retriever = build_retriever(chunks, method)
        by_id = {c["id"]: c for c in chunks}
        for case in golden["retrieval"]:
            ranked = retriever.search(case["question"], top_k=3)
            doc_ids = [by_id[cid]["doc_id"] for cid, _ in ranked]
            assert case["gold_doc"] in doc_ids, (
                f"{method}: gold {case['gold_doc']} not in top-3 for {case['id']}: {doc_ids}"
            )


def test_bm25_and_tfidf_agree_on_the_easiest_question():
    corpus, _ = _load()
    chunks = chunk_corpus(corpus, "passage")
    tops = []
    for method in ("tfidf", "bm25"):
        ranked = build_retriever(chunks, method).search(
            "What mainly causes ocean tides?", top_k=1)
        tops.append(ranked[0][0])
    assert tops[0] == tops[1] == "tides"


def test_sentence_chunking_still_finds_gold_doc():
    corpus, golden = _load()
    chunks = chunk_corpus(corpus, "sentence")
    retriever = build_retriever(chunks, "tfidf")
    by_id = {c["id"]: c for c in chunks}
    for case in golden["retrieval"][:4]:
        ranked = retriever.search(case["question"], top_k=5)
        doc_ids = [by_id[cid]["doc_id"] for cid, _ in ranked]
        assert case["gold_doc"] in doc_ids
