"""LLM-as-a-judge interface.

Two backends, one interface — ``judge(claim, evidence)`` returns
``{verdict, rationale, backend}`` with verdict in
``supported | unsupported | uncertain``.

- ``DemoJudge`` (default): deterministic, no API key, no network.
  It uses the faithfulness checker plus retrieval overlap, so it can
  be run in CI and in the sweep. It is *not* a real LLM judge and
  must not be cited as one.
- ``ApiJudge`` (optional): an OpenAI-compatible Chat Completions
  endpoint, configured only via environment variables
  (``RAGEVAL_API_KEY``, ``RAGEVAL_API_BASE``, ``RAGEVAL_MODEL``).
  It is not run by the default commands, and no key is ever committed.
  If it is not run, say so — do not report API-judge numbers you did
  not measure.
"""
import json
import os
import urllib.request

from .faithfulness import check_faithfulness


class DemoJudge:
    backend = "demo"

    def judge(self, claim, evidence):
        result = check_faithfulness(claim, evidence)
        if result["predicted_supported"]:
            verdict = "supported"
            rationale = (
                f"Overlap {result['overlap']} with no negation, role-order or "
                "missing-entity flag."
            )
        elif result["checks"]:
            verdict = "unsupported"
            rationale = (
                f"Faithfulness check(s) fired: {', '.join(result['checks'])}; "
                f"overlap was {result['overlap']}, capped score {result['score']}."
            )
        else:
            verdict = "uncertain"
            rationale = f"Overlap {result['overlap']} below the 0.5 threshold, no specific flag."
        return {"verdict": verdict, "rationale": rationale, "backend": self.backend,
                "score": result["score"], "checks": result["checks"]}


class ApiJudge:
    backend = "api"

    def __init__(self, api_key=None, base=None, model=None):
        self.api_key = api_key or os.environ.get("RAGEVAL_API_KEY")
        self.base = (base or os.environ.get("RAGEVAL_API_BASE",
                                           "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("RAGEVAL_MODEL", "gpt-4.1-mini")
        if not self.api_key:
            raise ValueError("RAGEVAL_API_KEY is not set; refusing to run the API judge.")

    def judge(self, claim, evidence):
        prompt = (
            "You are an LLM-as-a-judge for retrieval-grounded answers.\n"
            f"Evidence: {evidence}\nClaim: {claim}\n"
            "Reply with exactly one word on the first line: SUPPORTED, "
            "UNSUPPORTED, or UNCERTAIN, then one sentence of rationale."
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
        text = payload["choices"][0]["message"]["content"].strip()
        first, _, rest = text.partition("\n")
        verdict = first.strip().lower()
        if verdict not in {"supported", "unsupported", "uncertain"}:
            verdict = "uncertain"
        return {"verdict": verdict, "rationale": rest.strip() or text,
                "backend": self.backend, "model": self.model}


def get_judge():
    """Demo judge unless RAGEVAL_JUDGE=api is set (then API judge)."""
    if os.environ.get("RAGEVAL_JUDGE") == "api":
        return ApiJudge()
    return DemoJudge()
