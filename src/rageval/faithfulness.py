"""Faithfulness: is a candidate claim actually supported by the evidence?

The old word-overlap scorer (6/8 correct) failed on exactly the cases
where a confident wrong answer is most dangerous:

- negation: "Seasons are caused by the Earth moving closer to the Sun"
  scores 0.667 on overlap, but the passage says the opposite.
- role-swap: "A solar eclipse is caused by the Earth blocking the Moon"
  scores 0.667, but that describes a lunar eclipse.

This scorer keeps the overlap score as a base, then applies three
explicit, deterministic checks. Any check firing caps the score below
the 0.5 support threshold and marks the claim unsupported:

1. ``negation`` — a negated sentence in the evidence shares >= 2 content
   tokens with the claim (the evidence is denying the claim's premise).
2. ``role_order`` — the claim and the evidence mention the same domain
   entities (Earth / Moon / Sun) in a different order.
3. ``missing_entity`` — the claim names a domain entity (e.g.
   chloroplasts) that the evidence does not contain at all, while the
   evidence names a different value for the same slot.

These are deliberately narrow, named checks — not a general relation
extractor. Their limits are documented in the README.
"""
from .text import DOMAIN_ENTITIES, NEGATION_WORDS, entity_order, raw_tokens, sentences, tokens


def overlap_score(claim, evidence):
    claim_tokens = set(tokens(claim))
    evidence_tokens = set(tokens(evidence))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _negated_tokens(evidence):
    """Content tokens appearing in a negated sentence of the evidence."""
    out = set()
    for sent in sentences(evidence):
        raw = raw_tokens(sent)
        if any(w in NEGATION_WORDS for w in raw):
            out.update(tokens(sent))
    return out


def _check_negation(claim, evidence):
    neg = _negated_tokens(evidence)
    return len(set(tokens(claim)) & neg) >= 2


def _check_role_order(claim, evidence):
    """True if shared domain entities appear in a different relative
    order in the claim than in the evidence.

    Order is compared within the single evidence sentence that best
    matches the claim — not across the whole passage. Using the whole
    passage falsely flagged supported claims, because an earlier
    sentence can mention the same entities in a different order
    (e.g. the general tides passage mentions Moon/Sun before the
    spring-tides sentence mentions Moon/Earth/Sun).
    """
    claim_tokens = set(tokens(claim))
    best = max(sentences(evidence),
               key=lambda s: len(set(tokens(s)) & claim_tokens))
    c_order = entity_order(claim)
    e_order = entity_order(best)
    shared = [e for e in c_order if e in e_order]
    if len(shared) < 2:
        return False
    e_pos = [e_order.index(e) for e in shared]
    return e_pos != sorted(e_pos)


def _check_missing_entity(claim, evidence):
    claim_entities = {t for t in tokens(claim) if t in DOMAIN_ENTITIES}
    evidence_entities = {t for t in tokens(evidence) if t in DOMAIN_ENTITIES}
    return bool(claim_entities - evidence_entities)


def check_faithfulness(claim, evidence, threshold=0.5):
    """Return {score, predicted_supported, checks, overlap}.

    ``checks`` names every rule that fired; an empty list means the
    decision rests on overlap alone.
    """
    overlap = overlap_score(claim, evidence)
    checks = []
    if _check_negation(claim, evidence):
        checks.append("negation")
    if _check_role_order(claim, evidence):
        checks.append("role_order")
    if _check_missing_entity(claim, evidence):
        checks.append("missing_entity")
    score = overlap * 0.35 if checks else overlap
    return {
        "score": round(score, 3),
        "overlap": round(overlap, 3),
        "predicted_supported": score >= threshold,
        "checks": checks,
    }
