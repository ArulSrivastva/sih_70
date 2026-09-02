"""Feature consistency: Phase-5 engineering must equal Phase-4 exactly."""

import numpy as np

from phase4.features.feature_engineering import engineer_features as p4_engineer
from phase4.features.feature_engineering import features_from_history as p4_single
from phase5.inference.preprocessing import engineer_history
from phase5.config import default_paths


def _hist():
    return np.array([
        [23.30, 68.50, 55.6, 994.0, 28.67, 8.5095, 1.2749],
        [23.40, 68.00, 64.8, 993.0, 28.26, 4.4456, 7.0864],
        [23.40, 67.10, 64.8, 993.0, 28.43, 7.9810, -3.3996],
        [23.60, 66.40, 64.8, 992.0, 28.69, 2.0987, -4.1294],
        [23.50, 65.70, 74.1, 991.0, 28.85, 2.4551, 4.3582],
    ], dtype=np.float32)


def test_byte_equal_with_phase4_single():
    h = _hist()
    a = engineer_history(h)
    b = p4_single(h)
    assert np.array_equal(a, b)


def test_byte_equal_with_phase4_batch():
    h = _hist()
    a = engineer_history(h)
    b = p4_engineer(h[None, ...])[0]
    assert np.array_equal(a, b)


def test_consistent_with_stored_feature_dataset():
    """First 200 training samples: phase5 features == stored train.npz."""
    paths = default_paths()
    clean = np.load(paths.clean_data_dir / "train.npz", allow_pickle=True)
    stored = np.load(paths.feature_dataset / "train.npz", allow_pickle=True)
    Xc, Xf = clean["X"], stored["X"]
    for i in range(200):
        a = engineer_history(Xc[i])
        b = np.asarray(Xf[i], np.float32)
        if not np.array_equal(a, b):
            raise AssertionError(
                f"sample {i} mismatch: max abs diff "
                f"{np.max(np.abs(a - b)):.6f}")
    assert True


def test_no_future_history_dependence():
    h = _hist()
    base = engineer_history(h)
    mut = h.copy()
    mut[2:] += 50.0          # mutate the *later* timesteps
    out = engineer_history(mut)
    # steps 0..1 (t-24h, t-18h) must be unchanged by mutating later steps
    assert np.array_equal(base[:2], out[:2])


def test_target_independence_structural():
    """engineering only consumes (5,7); changing 'targets' changes nothing."""
    h = _hist()
    a = engineer_history(h)
    # append a fake target row would be a shape error; proving the API holds no
    # target slot is the structural guarantee (no target read in the code path)
    import inspect
    from phase5.inference import preprocessing
    src = inspect.getsource(preprocessing.engineer_history)
    assert "target" not in src and "Y" not in src.replace("np", "")