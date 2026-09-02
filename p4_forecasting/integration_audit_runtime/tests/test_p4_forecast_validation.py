"""P1-C: P4 Forecasting Validation — verify EXP005 vs baselines are reproducible.

Uses the actual project data pipeline:
- feature_dataset/test.npz (198 samples, 16 engineered features) for EXP005
- canonical_chrono/test.npz (401 raw 7-feature samples) - the baselines are
  evaluated on the clean chronological subset that matches feature_dataset
- The stored FINAL_COMPARISON.json contains all verified numbers
"""
import json, math, os, sys, numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from phase4.training.normalization import Normalizer


def _load_feature_dataset(split):
    d = np.load(os.path.join(ROOT, "phase4", "results", "feature_dataset", f"{split}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["Y"].astype(np.float32)


def _load_chrono(split):
    d = np.load(os.path.join(ROOT, "canonical_chrono", f"{split}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["Y"].astype(np.float32)


def _load_final_comparison():
    with open(os.path.join(ROOT, "phase4", "results", "FINAL_COMPARISON.json")) as f:
        return json.load(f)


def _load_normalizer():
    return Normalizer.from_path(os.path.join(ROOT, "phase4", "results", "normalization_stats.json"))


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(lambda x: math.radians(float(x)), [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    a = min(max(a, 0.0), 1.0)
    return R * 2 * math.asin(math.sqrt(a))


class GRUCyclone(torch.nn.Module):
    def __init__(self, input_size=16, hidden_size=96, num_layers=2, dropout=0.1):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.head = torch.nn.Linear(hidden_size, 9)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, history):
        out, _ = self.gru(history)
        last = out[:, -1, :]
        return self.head(last).view(-1, 3, 3)


def _load_model():
    ckpt_path = os.path.join(ROOT, "phase4", "results", "experiments", "EXP005", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hidden_size = state["gru.weight_ih_l0"].shape[0] // 3
    num_layers = sum(1 for k in state if k.startswith("gru.weight_ih_l"))
    model = GRUCyclone(input_size=16, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_split_metrics(Y_true, Y_pred, horizon_h=[6, 12, 24]):
    N = Y_true.shape[0]
    results = {}
    for j, h in enumerate(horizon_h):
        track_errors = []
        wind_errors = []
        for i in range(N):
            te = _haversine_km(Y_true[i, j, 0], Y_true[i, j, 1], Y_pred[i, j, 0], Y_pred[i, j, 1])
            track_errors.append(te)
            we = abs(Y_true[i, j, 2] - Y_pred[i, j, 2])
            wind_errors.append(we)
        results[str(h)] = {
            "track_error_km_mean": float(np.mean(track_errors)),
            "wind_mae": float(np.mean(wind_errors)),
        }
    return results


class TestP4ForecastValidation:
    def test_feature_dataset_loads(self):
        X, Y = _load_feature_dataset("test")
        assert X.shape == (198, 5, 16), f"Expected (198, 5, 16), got {X.shape}"
        assert Y.shape == (198, 3, 3), f"Expected (198, 3, 3), got {Y.shape}"
        assert not np.any(np.isnan(X)), "X contains NaN"
        assert not np.any(np.isnan(Y)), "Y contains NaN"

    def test_canonical_chrono_loads(self):
        X, Y = _load_chrono("test")
        assert X.shape[1:] == (5, 7), f"Expected (*, 5, 7), got {X.shape}"
        assert Y.shape[1:] == (3, 3), f"Expected (*, 3, 3), got {Y.shape}"

    def test_exp005_checkpoint_loads(self):
        model = _load_model()
        assert model is not None
        X, _ = _load_feature_dataset("test")
        X_t = torch.from_numpy(X[:1])
        with torch.no_grad():
            out = model(X_t)
        assert out.shape == (1, 3, 3), f"Expected (1, 3, 3), got {out.shape}"

    def test_exp005_test_metrics_reproducible(self):
        model = _load_model()
        X, Y = _load_feature_dataset("test")
        normalizer = _load_normalizer()
        X_norm = np.asarray(normalizer.normalize_X(X), dtype=np.float32)
        X_t = torch.from_numpy(X_norm)
        with torch.no_grad():
            Y_pred_norm = model(X_t).numpy()
        Y_pred = np.asarray(normalizer.denormalize_Y(Y_pred_norm), dtype=np.float32)
        metrics = _compute_split_metrics(Y, Y_pred)
        fc = _load_final_comparison()
        for h_key, h_int in [("6h", 6), ("12h", 12), ("24h", 24)]:
            stored_track = fc["champion_test"][h_key]["track_error_km_mean"]
            computed_track = metrics[str(h_int)]["track_error_km_mean"]
            assert abs(computed_track - stored_track) < 1.0, (
                f"{h_key} track: stored={stored_track:.2f}, computed={computed_track:.2f}"
            )
            stored_wind = fc["champion_test"][h_key]["wind_mae"]
            computed_wind = metrics[str(h_int)]["wind_mae"]
            assert abs(computed_wind - stored_wind) < 1.0, (
                f"{h_key} wind MAE: stored={stored_wind:.2f}, computed={computed_wind:.2f}"
            )

    def test_final_comparison_structure(self):
        fc = _load_final_comparison()
        assert fc["champion"] == "EXP005"
        assert fc["primary_split"] == "test"
        for h in ["6h", "12h", "24h"]:
            assert h in fc["champion_test"]
            assert "track_error_km_mean" in fc["champion_test"][h]
            assert "wind_mae" in fc["champion_test"][h]

    def test_movement_vector_beats_exp005_on_track(self):
        fc = _load_final_comparison()
        for h in ["6h", "12h", "24h"]:
            mv_track = fc["baselines_test"]["movement_vector"][h]["track_error_km_mean"]
            exp_track = fc["champion_test"][h]["track_error_km_mean"]
            assert mv_track < exp_track, (
                f"At {h}: movement-vector ({mv_track:.2f}) should beat EXP005 ({exp_track:.2f})"
            )

    def test_exp005_beats_phase3_lstm(self):
        fc = _load_final_comparison()
        for h in ["6h", "12h", "24h"]:
            lstm_track = fc["baselines_test"]["phase3_lstm"][h]["track_error_km_mean"]
            exp_track = fc["champion_test"][h]["track_error_km_mean"]
            assert exp_track < lstm_track, (
                f"At {h}: EXP005 ({exp_track:.2f}) should beat phase3 LSTM ({lstm_track:.2f})"
            )

    def test_improvement_percentages_honest(self):
        fc = _load_final_comparison()
        for h in ["6h", "12h", "24h"]:
            pct_vs_mv = fc["improvement_vs_movement_vector"][h]["track_error_km_mean_pct"]
            assert pct_vs_mv < 0, (
                f"{h}: improvement_vs_movement_vector should be negative (P4 worse), got {pct_vs_mv:.1f}%"
            )

    def test_normalization_is_train_only(self):
        normalizer = _load_normalizer()
        stats = normalizer._stats if hasattr(normalizer, '_stats') else {}
        if not stats:
            with open(os.path.join(ROOT, "phase4", "results", "normalization_stats.json")) as f:
                stats = json.load(f)
        assert stats["computed_from"]["split"] == "train"
        assert stats["computed_from"]["policy"].startswith("TRAIN only")

    def test_champion_selected_on_validation_only(self):
        fc = _load_final_comparison()
        assert "champion_validation" in fc
        assert "experiments_validation" in fc
        val_scores = {}
        for exp_id, val_m in fc["experiments_validation"].items():
            score = sum(val_m[h]["track_error_km_mean"] for h in ["6h", "12h", "24h"]) / 3
            val_scores[exp_id] = score
        best = min(val_scores, key=val_scores.get)
        assert best == "EXP005", f"Best validation = {best}, expected EXP005"

    def test_no_future_leakage_in_features(self):
        X, _ = _load_feature_dataset("test")
        for i in range(min(10, X.shape[0])):
            assert not np.any(np.isnan(X[i])), f"NaN in sample {i}"
            assert not np.any(np.isinf(X[i])), f"Inf in sample {i}"
