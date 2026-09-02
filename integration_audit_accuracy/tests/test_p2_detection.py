"""P2 detection assertions: reported metrics are reproduced, presence undemonstrated."""

import pytest


def test_reported_metrics_reproduced_exactly(p2):
    for k in p2["stored"]:
        assert abs(p2["stored"][k] - p2["recomputed"][k]) < 1e-12, k


def test_stored_values_match_reported(p2):
    assert p2["recomputed"]["pattern_accuracy"] == pytest.approx(0.7142857142857143, abs=1e-9)
    assert p2["recomputed"]["category_accuracy"] == pytest.approx(0.3333333333333333, abs=1e-9)


def test_stored_json_matches_reported_values(p2):
    assert p2["stored"]["pattern_f1"] == pytest.approx(0.6306522609, abs=1e-9)
    assert p2["stored"]["category_f1"] == pytest.approx(0.2305037957, abs=1e-9)


def test_presence_head_never_evaluated_degenerate(p2):
    # no negatives exist: every test label is "cyclone present"
    assert p2["presence_all_gt_positive"] is True


def test_structural_pattern_derived_not_annotated(p2):
    assert p2["pattern_derivation_bystand_ckt_agree"] > 0.99


def test_checkpoint_remains_mappable(p2):
    assert set(p2["pattern_to_idx"].keys()) == {"curved_band", "eye_visible", "shear_pattern"}
    assert len(p2["category_to_idx"]) == 7