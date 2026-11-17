import json
from pathlib import Path

from rageval.router import route
from rageval.text import tokens

DATA = Path(__file__).resolve().parents[1] / "data"


def _vocab():
    corpus = json.loads((DATA / "corpus.json").read_text())
    vocab = set()
    for doc in corpus:
        vocab.update(tokens(doc["text"]))
    return vocab


def test_router_retrieves_for_a_factual_question():
    decision = route("What mainly causes ocean tides?", _vocab())
    assert decision["action"] == "retrieve"
    assert "reason" in decision and "confidence" in decision


def test_router_answers_directly_for_a_greeting():
    decision = route("Hello, what can you do?", _vocab())
    assert decision["action"] == "answer_directly"


def test_router_logs_a_decision_for_every_input():
    for q in ["What is a solar eclipse?", "Hi", "Tell me about volcanoes"]:
        decision = route(q, _vocab())
        assert set(decision) >= {"action", "reason", "confidence", "overlap"}
        assert decision["action"] in {"retrieve", "answer_directly"}
