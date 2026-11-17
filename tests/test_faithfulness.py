import json
from pathlib import Path

from rageval.faithfulness import check_faithfulness, overlap_score

DATA = Path(__file__).resolve().parents[1] / "data"


def _by_id():
    corpus = json.loads((DATA / "corpus.json").read_text())
    return {d["id"]: d["text"] for d in corpus}


def test_overlap_only_fails_the_negation_case():
    """Reproduces the original failure, so the fix is tested, not assumed."""
    by_id = _by_id()
    claim = "Seasons are caused by the Earth moving closer to the Sun."
    assert overlap_score(claim, by_id["seasons"]) >= 0.5


def test_faithfulness_catches_the_negation_case():
    by_id = _by_id()
    claim = "Seasons are caused by the Earth moving closer to the Sun."
    result = check_faithfulness(claim, by_id["seasons"])
    assert result["predicted_supported"] is False
    assert "negation" in result["checks"]


def test_faithfulness_catches_the_role_swap_case():
    by_id = _by_id()
    claim = "A solar eclipse is caused by the Earth blocking the Moon."
    result = check_faithfulness(claim, by_id["eclipses"])
    assert result["predicted_supported"] is False
    assert "role_order" in result["checks"]


def test_faithfulness_catches_the_missing_entity_case():
    by_id = _by_id()
    claim = "Cellular respiration takes place mainly in chloroplasts."
    result = check_faithfulness(claim, by_id["cellular-respiration"])
    assert result["predicted_supported"] is False
    assert "missing_entity" in result["checks"]


def test_faithfulness_keeps_a_supported_claim():
    by_id = _by_id()
    claim = "Photosynthesis takes place mainly in chloroplasts."
    result = check_faithfulness(claim, by_id["photosynthesis"])
    assert result["predicted_supported"] is True
    assert result["checks"] == []
