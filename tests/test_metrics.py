import json
from pathlib import Path

from rageval.runner import evaluate_config, load_data


def test_metrics_are_deterministic_on_rerun():
    corpus, questions, claims = load_data()
    first = evaluate_config(corpus, questions, claims, "tfidf", "passage")
    second = evaluate_config(corpus, questions, claims, "tfidf", "passage")
    # Latency is a timing proxy and is excluded on purpose.
    for key in ("recall_at_1", "recall_at_3", "recall_at_5",
                "faithfulness_accuracy", "error_counts"):
        assert first["summary"][key] == second["summary"][key]


def test_ambiguous_questions_are_excluded_from_recall():
    corpus, questions, claims = load_data()
    result = evaluate_config(corpus, questions, claims, "tfidf", "passage")
    assert result["summary"]["n_ambiguous"] >= 4
    assert result["summary"]["n_questions"] == (
        result["summary"]["n_non_ambiguous"] + result["summary"]["n_ambiguous"])


def test_golden_retrieval_cases_pass_in_the_runner():
    corpus, questions, claims = load_data()
    golden = json.loads((Path(__file__).resolve().parents[1] / "data" / "golden.json").read_text())
    result = evaluate_config(corpus, questions, claims, "bm25", "passage")
    by_id = {r["id"]: r for r in result["questions"]}
    for case in golden["retrieval"]:
        assert by_id[case["id"]]["gold_doc"] in by_id[case["id"]]["ranked_docs"][:3]
