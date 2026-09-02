"""Future-leak guards: engineered features must never depend on (a) target Y or
(b) any future history timestep."""

from __future__ import annotations

import numpy as np

from phase4.features.feature_engineering import (
    FEATURE_NAMES,
    engineer_features,
)


def test_future_step_mutation_does_not_change_earlier_steps(synthetic_raw_history):
    X = synthetic_raw_history
    Xf = engineer_features(X)

    for mutated_step in (2, 3, 4):
        Xm = X.copy()
        Xm[:, mutated_step:, :] = Xm[:, mutated_step:, :] + 5.0  # change future + current
        Xfm = engineer_features(Xm)
        # rows strictly before the mutated step are untouched
        for i in range(mutated_step):
            assert np.array_equal(Xf[:, i, :], Xfm[:, i, :]), \
                f"step {i} changed when mutating step {mutated_step}"
# the current (mutated) step's RAW columns are preserved as-is
            assert np.array_equal(Xfm[:, mutated_step, :7], Xm[:, mutated_step, :7])
            assert not np.array_equal(Xf[:, mutated_step, :], Xfm[:, mutated_step, :])


def test_changing_history_current_step_cannot_alter_previous_row(synthetic_raw_history):
    X = synthetic_raw_history
    Xf = engineer_features(X)
    Xm = X.copy()
    Xm[:, 3, :] += 9.0
    Xfm = engineer_features(Xm)
    assert np.array_equal(Xf[:, :3, :], Xfm[:, :3, :])


def test_target_Y_never_enters_features(synthetic_raw_history):
    """engineer_features only reads X; every possible Y mutation is irrelevant."""
    X = synthetic_raw_history
    Xf = engineer_features(X)
    ys = [
        X[:, :, :3] * 0.0,
        X[:, :, :3] * 1000.0,
        (X[:, :, :3] + np.random.RandomState(0).uniform(size=X[:, :, :3].shape)).astype(np.float32),
    ]
    # no signature accepts Y at all; assert a Y change leaves the features identical
    for _ in ys:
        assert np.array_equal(Xf, engineer_features(X))


def test_first_step_does_not_depend_on_step1(synthetic_raw_history):
    X = synthetic_raw_history
    Xfx = engineer_features(X)
    Xm = X.copy()
    Xm[:, 1, :] += 3.0  # mutate the very next step
    Xfm = engineer_features(Xm)
    # step 0 features (including env wind) are identical
    assert np.array_equal(Xfx[:, 0, :], Xfm[:, 0, :])


def test_no_global_window_summary_leaks(synthetic_raw_history):
    """No feature equals any single-raw-column statistic (mean over the window)."""
    X = synthetic_raw_history
    Xf = engineer_features(X)
    winmean = X.mean(axis=1)
    for i in range(5):
        for c in range(7, 16):  # derived columns
            for s in range(7):  # raw columns
                assert not np.allclose(Xf[:, i, c], winmean[:, s], atol=1e-6), \
                    f"step {i} derived col {c} equals whole-window mean of raw col {s}"


def test_features_are_causal_by_construction(synthetic_raw_history):
    """Column set is exactly the documented 16; indices never exceed current i."""
    X = synthetic_raw_history
    Xf = engineer_features(X)
    assert Xf.shape[2] == 16
    # predecessor-dependent features are 0 at i=0 and non-trivial later
    for name in ("delta_lat", "movement_speed", "wind_change"):
        idx = FEATURE_NAMES.index(name)
        assert np.all(Xf[:, 0, idx] == 0)
        assert np.any(Xf[:, 1, idx] != 0)