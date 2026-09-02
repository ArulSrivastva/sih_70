"""P4 forecasting: prefixes, EXP005 test, baselines, selection hygiene, EXP006 anomaly."""


def test_canonical_chrono_prefixes(p4):
    for key, meta in p4["canonical_chrono_sha256_prefixes"].items():
        assert meta["match"] is True, f"{key} prefix mismatch"


def test_exp005_test_metrics_reproduced_within_tolerance(p4):
    pdiff = p4["EXP005_stored_test_vs_recomputed_max_diff"]
    for h in ["6h", "12h", "24h"]:
        assert pdiff[h] < 1e-4, h


def test_baselines_reproduced(p4):
    rc = p4["baselines_recomputed"]
    st = p4["baselines_stored"]
    key_for = {"persistence": "persistence", "movement_vector": "movement"}
    for m in ["persistence", "movement_vector"]:
        for h in ["6h", "12h", "24h"]:
            assert abs(rc[key_for[m]][h] - st[m][h]) < 1e-6, (m, h)


def test_champion_loses_to_movement_vector(p4):
    assert p4["champion_loses_to_movement_vector_all_horizons"] is True


def test_champion_vs_persistence_at_6h(p4):
    assert p4["champion_loses_to_persistence_6h_win_12h_24h"] is True
    c = p4["champion_vs_baselines"]
    assert c["6h"]["EXP005"] > c["6h"]["persistence"]
    assert c["12h"]["EXP005"] < c["12h"]["persistence"]
    assert c["24h"]["EXP005"] < c["24h"]["persistence"]


def test_val_primary_is_mean_of_val_track(p4):
    ep = p4["registry_experiments"]["EXP005"]
    mean = sum(ep["val_track"]) / 3.0
    assert abs(mean - ep["val_primary"]) < 1e-6


def test_rank_matches_champion(p4):
    assert p4["rank_order_by_val_primary"][0] == "EXP005"
    assert p4["champion_event"]["experiment_id"] == "EXP005"


def test_exp006_registry_anomaly_flagged(p4):
    assert p4["EXP006_EXP003_metrics_near_identical_flag"] is True


def test_test_population_note_present(p4):
    assert "feature_dataset/test.npz (198 rows" in p4["test_population_note"]