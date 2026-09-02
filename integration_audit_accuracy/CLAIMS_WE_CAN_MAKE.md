# Claims We Can Make

Every claim below is grounded in a read-only independent recomputation or direct file inspection performed during this audit.

## VERIFIED (reproduced bit-close on delivered artifacts)

1. **Data is internally consistent and well-formed**: master 5,481 rows / 151 cyclones; dup-free; splits disjoint (105/22/24); ERA5 joined 100% (5,481/5,481); physical ranges clean; category↔wind mapping exact.
2. **Splits are random (not chronological)**: test first timestamp 2013-05-29 lies inside the train era.
3. **P2 detection pattern/category metrics reproduce exactly** (0.0 diff on all 8 metrics) from the shipped checkpoint.
4. **P3 image-model metrics reproduce exactly** (acc 38.1, macro-F1 0.2076, wind MAE 109.81, RMSE 118.39).
5. **P4 canonical chronology contract holds**: SHA256 prefixes match expected (train `df70303e…`, val `48cf065d…`, test `89e9c2e2…`).
6. **P4 EXP005 test metrics reproduce** to ≤1e-4 (track mean 91.43 / 119.76 / 188.24 km; wind MAE 6.68 / 9.46 / 16.11 km/h).
7. **P4 baselines reproduce to machine precision** (persistence 64.69 / 123.94 / 227.33; movement-vector 38.05 / 80.50 / 180.66 km).
8. **Selection hygiene is genuine**: champion picked on validation only; test used once (only EXP005 has test_results.json).
9. **Phase-6 API works offline** (verified live): /health, /model, POST /forecast, /forecast/compare; strict validation (422 on 4-step, missing field, out-of-range lat).
10. **Normalization is train-only** (`computed_from: train`), step-0 zero-fill, causally clean feature windows.

## VERIFIED BUT SUBJECT TO THE FOLLOWING LIMITS (say with the caveat or it is misleading)

- P2 pattern/category scores are on **synthetic labels** (wind-derived), 21 frames, no presence/bbox evaluation → use only as “classifier on proxy labels”, never “detection”.
- P3 image scores share the same 21-frame test; **same-storm frames leak across splits** (e.g., 36.jpg/36(3).jpg), so scores are optimistic.
- P4 champion **does not beat the movement-vector baseline on track at any horizon**, and loses to persistence at 6h. It wins vs persistence at 12h/24h and vs phase-3 LSTM. Wind forecasts are its defensible value.
- Reported test/baseline numbers are defined on **phase4 feature_dataset (198 rows)**, not the delivered canonical_chrono test.npz (401 rows); a fresh run on canonical_chrono will not reproduce them.

## NOT SOURCED / NOT VERIFIABLE HERE (do not present as established)

- Any “future prediction / operational forecasting” claim (random split, small test).
- The multi-source-vs-image “advantage” (different populations; tabular model NOT_RUN under audit env).
- Bounding-box / localization quality, false-positive rates (none testable — no negatives).
- Phase-3 LSTM test numbers (no weights delivered; cited from stored JSON only).
- “Completeness 100.0%” wording (conflicts with documented sst missing 1,538).