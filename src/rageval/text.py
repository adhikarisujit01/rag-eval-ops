"""Shared text helpers: tokenisation and sentence splitting.

Kept in one place so chunking, retrieval, reranking and faithfulness
all use the same definitions — a common source of silent drift in
evaluation harnesses.
"""
import re

STOPWORDS = set(
    "a an the is are was were be been being what why how when where which "
    "who whom whose does do did done not no yes of to in on at by for with "
    "and or as from it its this that than then there here can could should "
    "would will shall may might".split()
)

# Domain nouns where a wrong value is a factual error, not a paraphrase.
# Used by the faithfulness checker: claiming one of these when the
# evidence names a different one is a contradiction, not a wording choice.
DOMAIN_ENTITIES = {
    "moon", "earth", "sun", "chloroplasts", "mitochondria", "leaves",
    "sunlight", "oxygen", "glucose", "carbon", "dioxide",
}

NEGATION_WORDS = {"not", "no", "never", "without", "cannot", "n't"}


def tokens(text):
    """Lower-case word tokens, stopwords removed."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOPWORDS]


def raw_tokens(text):
    """Lower-case word tokens, stopwords kept (for negation/order checks)."""
    return re.findall(r"[a-z]+", text.lower())


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def entity_order(text):
    """Domain entities in order of first appearance."""
    seen = []
    for t in raw_tokens(text):
        if t in DOMAIN_ENTITIES and t not in seen:
            seen.append(t)
    return seen
