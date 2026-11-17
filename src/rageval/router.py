"""Agentic router: retrieve, or answer directly?

A deliberately small, transparent router — the point is the *logged
decision*, not cleverness. It decides, for each incoming question:

- ``retrieve`` — a factual question whose answer should come from the
  corpus (wh-questions, yes/no questions about a corpus topic).
- ``answer_directly`` — a greeting, a capability question, or a
  question with no overlap at all with the corpus vocabulary.

Every decision is returned as a dict with ``action``, ``reason`` and
``confidence`` so the reasoning is inspectable in logs and in tests.
No LLM call is made here.
"""
from .text import tokens

GREETINGS = {"hi", "hello", "hey", "good", "morning", "afternoon"}
CAPABILITY_PHRASES = {"what can you do", "who are you", "help", "how do you work"}
QUESTION_WORDS = {"what", "why", "how", "which", "when", "where", "who",
                  "does", "do", "did", "is", "are", "was", "were", "can"}


def route(question, corpus_vocab, threshold=1):
    """corpus_vocab: set of content tokens appearing in the corpus."""
    q = question.strip().lower()
    q_tokens = set(tokens(question))
    overlap = len(q_tokens & corpus_vocab)
    if any(p in q for p in CAPABILITY_PHRASES) or q_tokens & GREETINGS and overlap == 0:
        return {"action": "answer_directly",
                "reason": "greeting or capability question; no retrieval needed",
                "confidence": 0.9, "overlap": overlap}
    first = q.split()[0] if q.split() else ""
    if first in QUESTION_WORDS and overlap >= threshold:
        return {"action": "retrieve",
                "reason": f"factual question with {overlap} corpus token(s) in common",
                "confidence": 0.8, "overlap": overlap}
    if overlap >= threshold:
        return {"action": "retrieve",
                "reason": f"content overlap with corpus ({overlap} token(s))",
                "confidence": 0.6, "overlap": overlap}
    return {"action": "answer_directly",
            "reason": "no overlap with corpus vocabulary; retrieval would add noise",
            "confidence": 0.7, "overlap": overlap}
