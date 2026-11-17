"""Evaluation runner: one config, and the full sweep.

    python -m rageval run --retriever tfidf --chunking passage
    python -m rageval run --retriever bm25 --chunking sentence --rerank
    python -m rageval sweep          # all chunking x retriever configs
    python -m rageval check          # regression check vs results/baseline.json

Recall is computed over non-ambiguous questions only. Latency is a
retrieval-time proxy (mean ms per query, 3 repeats of the full set),
not a production latency claim — it varies run to run and is not
asserted in tests.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .chunking import chunk_corpus
from .faithfulness import check_faithfulness, overlap_score
from .judge import DemoJudge
from .metrics import classify, error_counts, extractive_answer, recall_at_k
from .reranking import rerank
from .retrieval import build_retriever
from .router import route
from .text import tokens

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_data():
    corpus = json.loads((DATA / "corpus.json").read_text())
    questions = json.loads((DATA / "questions.json").read_text())
    claims = json.loads((DATA / "faithfulness_claims.json").read_text())
    return corpus, questions, claims


def evaluate_config(corpus, questions, claims, retriever_name="tfidf",
                    chunking="passage", use_rerank=False, repeats=3):
    chunks = chunk_corpus(corpus, chunking)
    chunks_by_id = {c["id"]: c for c in chunks}
    by_doc = {d["id"]: d["text"] for d in corpus}
    retriever = build_retriever(chunks, retriever_name)

    # Latency proxy: repeat the full retrieval set, take the mean.
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        for q in questions:
            retriever.search(q["question"], top_k=5)
        latencies.append((time.perf_counter() - start) / len(questions) * 1000)
    mean_latency_ms = round(sum(latencies) / len(latencies), 3)

    rows = []
    for q in questions:
        ranked = retriever.search(q["question"], top_k=5)
        if use_rerank:
            ranked = rerank(q["question"], ranked, chunks_by_id, top_k=5)
        # Ranked doc ids, deduplicated, in order.
        ranked_docs = []
        for cid, _ in ranked:
            doc_id = chunks_by_id[cid]["doc_id"]
            if doc_id not in ranked_docs:
                ranked_docs.append(doc_id)
        top_text = by_doc[ranked_docs[0]]
        answer = extractive_answer(top_text, q["question"])
        row = {"id": q["id"], "question": q["question"], "gold_doc": q["gold_doc"],
               "ambiguous": q.get("ambiguous", False),
               "gold_keywords": q["gold_keywords"],
               "top1": ranked_docs[0], "ranked_docs": ranked_docs,
               "answer": answer}
        row["error"] = classify(
            {"gold_doc": q["gold_doc"], "gold_keywords": q["gold_keywords"],
             "ambiguous": q.get("ambiguous", False), "ranked_docs": ranked_docs},
            top_text, answer)
        rows.append(row)

    # Faithfulness: new scorer vs overlap-only, on the same claims.
    judge = DemoJudge()
    claim_rows = []
    for c in claims:
        evidence = by_doc[c["doc_id"]]
        new = check_faithfulness(c["claim"], evidence)
        old_pred = overlap_score(c["claim"], evidence) >= 0.5
        verdict = judge.judge(c["claim"], evidence)
        claim_rows.append({
            "claim": c["claim"], "supported": c["supported"],
            "doc_id": c["doc_id"],
            "score": new["score"], "overlap": new["overlap"],
            "checks": new["checks"], "predicted_supported": new["predicted_supported"],
            "correct": new["predicted_supported"] == c["supported"],
            "overlap_only_predicted_supported": old_pred,
            "overlap_only_correct": old_pred == c["supported"],
            "judge_verdict": verdict["verdict"],
        })

    summary = {
        "config": {"retriever": retriever_name, "chunking": chunking, "rerank": use_rerank},
        "n_questions": len(questions),
        "n_non_ambiguous": sum(1 for r in rows if not r["ambiguous"]),
        "n_ambiguous": sum(1 for r in rows if r["ambiguous"]),
        "recall_at_1": recall_at_k(rows, 1),
        "recall_at_3": recall_at_k(rows, 3),
        "recall_at_5": recall_at_k(rows, 5),
        "mean_retrieval_ms_per_query": mean_latency_ms,
        "error_counts": error_counts(rows),
        "faithfulness_accuracy": round(
            sum(c["correct"] for c in claim_rows) / len(claim_rows), 3) if claim_rows else 0.0,
        "faithfulness_overlap_only_accuracy": round(
            sum(c["overlap_only_correct"] for c in claim_rows) / len(claim_rows), 3) if claim_rows else 0.0,
        "n_faithfulness_claims": len(claim_rows),
    }
    return {"summary": summary, "questions": rows, "faithfulness_claims": claim_rows}


def run_sweep():
    corpus, questions, claims = load_data()
    configs = []
    for chunking in ["passage", "sentence", "window"]:
        for retriever_name in ["tfidf", "bm25"]:
            configs.append((retriever_name, chunking, False))
    # One reranked variant, on the strongest-looking base config.
    configs.append(("bm25", "passage", True))
    results = []
    for retriever_name, chunking, use_rerank in configs:
        result = evaluate_config(corpus, questions, claims, retriever_name, chunking, use_rerank)
        results.append(result["summary"])
        print(f"{retriever_name:6} {chunking:9} rerank={use_rerank!s:5} "
              f"R@1={result['summary']['recall_at_1']} "
              f"R@3={result['summary']['recall_at_3']} "
              f"R@5={result['summary']['recall_at_5']} "
              f"faith={result['summary']['faithfulness_accuracy']} "
              f"{result['summary']['mean_retrieval_ms_per_query']}ms/q")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sweep.json").write_text(json.dumps(results, indent=2))
    return results


def write_results_md(result):
    s = result["summary"]
    lines = [
        "# Results (generated by `python -m rageval run`)",
        "",
        f"- Config: {s['config']}",
        f"- Questions: {s['n_questions']} ({s['n_non_ambiguous']} scored, "
        f"{s['n_ambiguous']} ambiguous, excluded from recall)",
        f"- Recall@1: {s['recall_at_1']}, Recall@3: {s['recall_at_3']}, "
        f"Recall@5: {s['recall_at_5']}",
        f"- Mean retrieval time: {s['mean_retrieval_ms_per_query']} ms/query "
        "(latency proxy, varies run to run)",
        f"- Error counts: {s['error_counts']}",
        f"- Faithfulness accuracy: {s['faithfulness_accuracy']} "
        f"({s['n_faithfulness_claims']} claims); overlap-only: "
        f"{s['faithfulness_overlap_only_accuracy']}",
        "",
        "## Per-question errors",
    ]
    for r in result["questions"]:
        lines.append(f"- {r['id']} [{r['error']}] top1={r['top1']} gold={r['gold_doc']} — {r['question']}")
    lines += ["", "## Faithfulness claims"]
    for c in result["faithfulness_claims"]:
        mark = "OK" if c["correct"] else "MISS"
        lines.append(f"- [{mark}] supported={c['supported']} score={c['score']} "
                     f"checks={c['checks']} — {c['claim']}")
    return "\n".join(lines) + "\n"


def cmd_run(args):
    corpus, questions, claims = load_data()
    result = evaluate_config(corpus, questions, claims, args.retriever,
                             args.chunking, args.rerank)
    RESULTS.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else RESULTS / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    (RESULTS / "results.md").write_text(write_results_md(result))
    print(json.dumps(result["summary"], indent=2))
    return 0


def cmd_sweep(_args):
    run_sweep()
    return 0


def cmd_check(_args):
    baseline_path = RESULTS / "baseline.json"
    if not baseline_path.exists():
        print("No baseline found at results/baseline.json — run `python -m rageval run` first.")
        return 2
    baseline = json.loads(baseline_path.read_text())
    corpus, questions, claims = load_data()
    result = evaluate_config(corpus, questions, claims,
                             baseline["config"]["retriever"],
                             baseline["config"]["chunking"],
                             baseline["config"].get("rerank", False))
    current = result["summary"]["recall_at_3"]
    expected = baseline["recall_at_3"]
    print(f"Baseline Recall@3: {expected} | Current Recall@3: {current}")
    if current < expected:
        print(f"REGRESSION: Recall@3 dropped below baseline ({current} < {expected}).")
        return 1
    print("OK: no regression vs baseline.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rageval")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Run one evaluation config")
    p_run.add_argument("--retriever", default="tfidf", choices=["tfidf", "bm25"])
    p_run.add_argument("--chunking", default="passage", choices=["passage", "sentence", "window"])
    p_run.add_argument("--rerank", action="store_true")
    p_run.add_argument("--output", default=None)
    p_run.set_defaults(func=cmd_run)
    p_sweep = sub.add_parser("sweep", help="Run the chunking x retriever sweep")
    p_sweep.set_defaults(func=cmd_sweep)
    p_check = sub.add_parser("check", help="Regression check vs results/baseline.json")
    p_check.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
