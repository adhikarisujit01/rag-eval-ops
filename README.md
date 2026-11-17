# rag-eval-ops

A small RAG evaluation harness with LLMOps regression testing — built to
show, with numbers, how retrieval, chunking, reranking and faithfulness
checking actually behave when the corpus contains hard near-duplicates.

**Status:** working portfolio prototype. Toy corpus only, standard library
only, no API key needed. **Not affiliated with Halfspace, Faculty, or
Accenture.** No production deployment, no real embeddings, no vector
database, and no real-LLM results are claimed here.

## What it is

An end-to-end, runnable demonstration of the exact loop the Halfspace
Generative AI Solutions role describes: build a RAG system, evaluate it
with a golden dataset, catch regressions in CI, and explain the
trade-offs honestly — including where the system fails.

It expands an earlier grounded-RAG project (10 passages, 12 questions,
Recall@1/@3 = 1.0, faithfulness 6/8) by doing the two things that
project's README called for next: adding near-duplicate passages to
create real retrieval misses, and replacing word-overlap faithfulness
with a relation-aware check.

## Halfspace mapping

| Halfspace requirement | Where it lives in this repo |
|---|---|
| RAG architectures and retrieval optimisation | `src/rageval/retrieval.py` — TF-IDF + BM25, both stdlib-only; `src/rageval/chunking.py` — passage / sentence / overlapping-window chunking |
| Retrieval optimisation, chunking, embeddings, vector databases | Chunking + retriever sweep in `src/rageval/runner.py` (`python -m rageval sweep`); embeddings/vector DB are a stated *limitation* — see below |
| LLMOps: evaluation frameworks, regression testing, monitoring, observability | `data/golden.json` (golden dataset), `tests/`, `results/baseline.json`, `python -m rageval check`, `.github/workflows/ci.yml` |
| LLM-as-a-judge | `src/rageval/judge.py` — deterministic demo judge (default, no key) + optional OpenAI-compatible API judge behind env vars |
| Agentic AI systems, agent architectures, orchestration | `src/rageval/router.py` — a small, transparent retrieve-vs-answer-directly router with a logged decision for every question |
| Strong Python engineering, testing, CI/CD, performance awareness | `tests/` (14 tests), CI on Python 3.10/3.11/3.12, latency proxy reported per config in the sweep |
| Turning ambiguous requirements into a scoped solution, explaining trade-offs | `results/` + the Results section below; ambiguous questions are flagged and excluded from recall, never counted as wins |
| Prototype-to-production experience (Azure/AWS/GCP) | Not claimed here — see Limitations |

## Quick start

```bash
python -m pip install -e .        # no runtime dependencies
python -m pytest -q                # 14 tests
python -m rageval run --retriever tfidf --chunking passage
python -m rageval sweep             # chunking x retriever sweep
python -m rageval check            # fails if Recall@3 drops below baseline
```

Try a different config:

```bash
python -m rageval run --retriever bm25 --chunking sentence --rerank
```

## Measured results

Run on 2026-11-17, Python 3.12, from a clean checkout of this directory.
26 questions (20 scored, 6 ambiguous — excluded from recall), 20 passages,
12 faithfulness claims. Recall is over non-ambiguous questions only.

### Retrieval sweep

| Retriever | Chunking | Rerank | Recall@1 | Recall@3 | Recall@5 | Mean retrieval ms/query* |
|---|---|---|---|---|---|---|
| TF-IDF | passage | no | 0.900 | 1.000 | 1.000 | 0.101 |
| BM25 | passage | no | 0.900 | 1.000 | 1.000 | 0.030 |
| TF-IDF | sentence | no | 0.700 | 1.000 | 1.000 | 0.233 |
| BM25 | sentence | no | 0.850 | 1.000 | 1.000 | 0.054 |
| TF-IDF | window | no | 0.850 | 1.000 | 1.000 | 0.107 |
| BM25 | window | no | 0.850 | 1.000 | 1.000 | 0.033 |
| BM25 | passage | yes | 0.900 | 1.000 | 1.000 | 0.029 |

\* Latency is a retrieval-time proxy (mean ms/query, 3 repeats of the
full set on this machine). It varies run to run and is not a production
latency claim.

**What the hard negatives changed vs the old 1.0:** the old corpus had
Recall@1 = Recall@3 = 1.0 — a ceiling effect, because every topic was
distinct. With near-duplicates (solar vs lunar eclipse, spring vs neap
tides, photosynthesis vs cellular respiration, shield vs stratovolcano),
Recall@1 drops to 0.70–0.90 depending on config: the gold passage is
still in the top 3 every time (Recall@3 = 1.0 in all 7 configs), but a
near-duplicate now wins rank 1 in 2 of 20 scored questions at passage
level — q13 (lunar eclipse, beaten by the general eclipses passage)
and q16 (neap tides, beaten by the general tides passage). Sentence
chunking is worst on Recall@1 (TF-IDF 0.700) and on grounding (6
grounding failures vs 2 at passage level): finer chunks lose the
context that disambiguates the pairs. Reranking did not change any
Recall number on this toy corpus — reported, not hidden.

Error counts, default config (TF-IDF, passage): 18 correct, 0 retrieval
misses, 2 grounding failures, 6 ambiguous. The 2 grounding failures
are extractive-answer failures, not retrieval failures: q17 (cellular
respiration — the answer picks the glucose/energy sentence, not the
mitochondria sentence) and q19 (chemosynthesis — the answer picks the
"without sunlight" sentence, not the chemical/sulfur sentence). That is
a real limitation of sentence-overlap answer extraction, and it is the
next thing to fix.

### Faithfulness

| Scorer | Accuracy (12 claims) |
|---|---|
| Word-overlap only (old scorer) | 0.667 (8/12) |
| This repo: overlap + negation / role-order / missing-entity checks | 0.917 (11/12) |

The new scorer now catches the two original failures: the negation
case ("Seasons are caused by the Earth moving closer to the Sun" —
evidence says the opposite) and the role-swap case ("A solar eclipse
is caused by the Earth blocking the Moon" — that is a lunar eclipse),
plus a missing-entity case (cellular respiration in chloroplasts —
evidence says mitochondria). The one remaining miss is also reported:
"Spring tides occur when the Moon and Sun pull at right angles" is
wrongly predicted supported (overlap 0.75, no check fires) — the
condition-swap is not caught by any of the three checks. That is a
known gap, not a hidden one.

The demo LLM-as-a-judge agrees with the new scorer on 11/12 claims
(the same spring-tides claim is its only disagreement with the gold
labels). The demo judge is deterministic, not a real LLM — do not cite
it as LLM-judge evidence.

## Data

All passages and questions are small toy data written for this project
in `data/corpus.json` (20 passages) and `data/questions.json`
(26 questions, 6 flagged ambiguous). They are **not a benchmark** and
not real-world data. No copyrighted text, no proprietary data.

Hard near-duplicate pairs added vs the original project: solar vs
lunar eclipse, spring vs neap tides, photosynthesis vs cellular
respiration, shield vs stratovolcano, plus chemosynthesis, adaptation
and new-moon passages that share vocabulary with their neighbours.

## Optional real-LLM judge

Not run for the numbers above. To run it:

```bash
cp .env.example .env   # then fill in your key; .env is gitignored, never commit it
export RAGEVAL_JUDGE=api
export RAGEVAL_API_KEY=...          # OpenAI-compatible key
export RAGEVAL_API_BASE=https://api.openai.com/v1   # optional
export RAGEVAL_MODEL=gpt-4.1-mini  # optional
python -m rageval run --retriever bm25 --chunking passage
```

The API backend uses `urllib` only (no extra dependency) and a
temperature of 0. If `RAGEVAL_JUDGE` is not set to `api`, the demo
judge is used and no network call is made.

## Regression testing (LLMOps)

- `data/golden.json` — 8 retrieval cases + 4 faithfulness cases that
  must keep passing, including the negation and role-swap cases.
- `results/baseline.json` — default config (TF-IDF, passage):
  Recall@3 = 1.0, Recall@1 = 0.9, faithfulness = 0.917.
- `python -m rageval check` — re-runs the baseline config and exits
  1 if Recall@3 drops below baseline. Verified: passes on the current
  code, and exits 1 when the baseline is artificially raised.
- `.github/workflows/ci.yml` — pytest on Python 3.10, 3.11 and 3.12,
  then the regression check.

## Limitations

- Toy corpus (20 passages, 26 questions): every Recall number here is
  a demonstration on a tiny dataset, not a benchmark result. Recall@3
  = 1.0 is still a ceiling effect at top-3 — the hard negatives only
  bite at rank 1.
- No embeddings and no vector database in this version. Retrieval is
  TF-IDF / BM25 over token overlap. Adding embeddings (and a real
  vector store) is the obvious next step, and it is not claimed as done.
- No cloud deployment (Azure/AWS/GCP), no production system, no
  real-LLM evaluation run. The API judge exists as code; it was not
  run for these results.
- The faithfulness checks are narrow, named rules (negation, entity
  role-order within the best-matching sentence, missing domain entity).
  They are not a general relation extractor, and the spring-tides
  condition-swap miss shows their limit.
- The agentic router is a small rule-based router, not an LLM agent.
  Its value here is the logged, testable decision — not autonomy.

## Related work

- **CulturaEval** (github.com/adhikarisujit01/CulturaEval-Cultural-Reasoning-LLMs) —
  LLM interpretation evaluation on culturally situated texts: 16
  GPT-4.1-mini responses, two independent reviewers.
- **enterprise-ai-control-plane** — Streamlit governance prototype for
  enterprise AI portfolio prioritisation and agentic-AI controls.
- **grounded-rag-evaluation** — the original project this expands:
  TF-IDF retrieval + extractive answers + word-overlap faithfulness on
  10 passages / 12 questions.

## Author

Sujit Adhikari — AI/NLP Engineer. MS Business Analytics, Golden Gate
University. This repo is a portfolio project built for a Generative AI
Solutions application; it is not affiliated with, endorsed by, or
representative of Halfspace, Faculty, or Accenture.
