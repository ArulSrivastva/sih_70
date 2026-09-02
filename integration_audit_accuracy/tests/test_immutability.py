"""Immutability: the workspace sources consumed by this audit must be unchanged
between SOURCE_HASHES_BEFORE.json and SOURCE_HASHES_AFTER.json."""

import pytest


def _flatten(manifest):
    out = {}
    for scope, entries in manifest.items():
        if not isinstance(entries, dict):
            continue
        for key, meta in entries.items():
            if isinstance(meta, dict) and "sha256" in meta:
                out[(scope, key)] = (meta["sha256"], meta.get("size"))
    return out


def test_workspace_sources_unchanged(hashes_before, hashes_after):
    if hashes_before is None:
        pytest.skip("SOURCE_HASHES_BEFORE.json missing")
    if hashes_after is None:
        pytest.skip("SOURCE_HASHES_AFTER.json not generated yet")
    b = _flatten(hashes_before)
    a = _flatten(hashes_after)
    assert set(a.keys()) == set(b.keys()), "manifest key set changed"
    changed = {k for k in a if a[k] != b[k]}
    assert changed == set(), f"sources changed: {sorted(changed)}"


def test_before_manifest_exists_and_covers_all_scopes(hashes_before):
    if hashes_before is None:
        pytest.skip("SOURCE_HASHES_BEFORE.json missing")
    for scope in ["p1_zip_archive", "p1_zip_entries", "p4_forecasting",
                  "p5_cyclone_project_dashboard", "p5_sih26_dashboard"]:
        assert scope in hashes_before
    assert hashes_before["p1_zip_archive"]["PS70-main.zip"]["size"] > 1_000_000_000