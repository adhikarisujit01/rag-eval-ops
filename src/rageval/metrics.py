"""Metrics and the error taxonomy.

Recall@k is computed over *non-ambiguous* questions only — ambiguous
items are flagged, reported separately, and never counted as wins.
That is the same rule as the original grounded-RAG project, and it
matters more now that the corpus has hard near-duplicates.

Error taxonomy, applied in order:
    retrieval_miss -> ambiguous -> grounding_failure -> correct
"""
from .text import sentences, tokens


def recall_at_k(rows, k):
    """rows: per-question dicts with gold_doc and a ranked list of doc ids."""
    scored = [r for r in rows if not r.get("ambiguous")]
    if not scored:
        return 0.0
    hits = sum(1 for r in scored if r["gold_doc"] in r["ranked_docs"][:k])
    return round(hits / len(scored), 3)


def extractive_answer(top_text, question):
    qset = set(tokens(question))
    return max(sentences(top_text),
               key=lambda s: len(set(tokens(s)) & qset))


def classify(question_row, top_text, answer):
    """Return the error label for one question."""
    if question_row["gold_doc"] not in question_row["ranked_docs"][:3]:
        return "retrieval_miss"
    if question_row.get("ambiguous"):
        return "ambiguous"
    hits = [k for k in question_row["gold_keywords"] if k in answer.lower()]
    need = 2 if len(question_row["gold_keywords"]) >= 2 else 1
    return "correct" if len(hits) >= need else "grounding_failure"


def error_counts(rows):
    labels = ["correct", "retrieval_miss", "grounding_failure", "ambiguous"]
    return {e: sum(1 for r in rows if r["error"] == e) for e in labels}
