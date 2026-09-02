# P2 Detection Audit

**Status: PASS_ON_REPORTED_METRICS / NOT_VERIFIABLE_AS_DETECTION.**

## What was verified

1. **Reported metrics are reproducible.** We independently ran the delivered `model_weights.pt` (MobileNetV3-small `CycloneDetector`, ImageNet-pretrained backbone) on the 21 test images with the exact shipped transform (`Resize((224,224))` + `ToTensor`), then recomputed the same sklearn metrics:

| Metric | Stored (`detection_metrics.json`) | Independent rerun | Diff |
|---|---|---|---|
| pattern_accuracy | 0.7142857143 | 0.7142857143 | 0.0 |
| pattern_precision (weighted) | 0.7523809524 | 0.7523809524 | 0.0 |
| pattern_recall (weighted) | 0.7142857143 | 0.7142857143 | 0.0 |
| pattern_f1 (weighted) | 0.6306522609 | 0.6306522609 | 0.0 |
| category_accuracy | 0.3333333333 | 0.3333333333 | 0.0 |
| category_precision (weighted) | 0.2566137566 | 0.2566137566 | 0.0 |
| category_recall (weighted) | 0.3333333333 | 0.3333333333 | 0.0 |
| category_f1 (weighted) | 0.2305037957 | 0.2305037957 | 0.0 |

Checkpoint keys verified: `model_state_dict`, `pattern_to_idx` (3), `category_to_idx` (7). Label maps are built from the **combined** train+val+test frame list at train time (a minor undisclosed practice; test labels never influence training, but the class index space is defined using test-set categories).

## 2. What this model actually measures

- **There is no detection problem here.** All 133 rows are positive (`cyclone_detected=True`), all bounding boxes are the single constant `[420, 190, 600, 370]`. The presence head is defined but **never evaluated** (not scored in `evaluate.py`, and there are no negatives to predict).
- `structural_pattern` and `category` target labels are **values derived from wind speed by fixed thresholds** (synthetic), not structural annotations by a meteorologist.
- Therefore the correct interpretation is: *a single-image classifier over model-derived labels on 21 test frames*, achieving pattern acc 71.4% / category acc 33.3% (7-way chance = 14.3%). **No claim about "detecting cyclones", locating them, or structural validity can be supported.**

## 3. Integration constraints

- Runtime contract `src/detection/inference.py`: `load_cyclone_detector()` / `detect_cyclone(image)`. The P5 dashboard requires a `/api/detect` endpoint with detection output; the delivered API exposes **no detection path at all** (see `API_COMPATIBILITY.md`). Consuming P2 in the app would require writing a new endpoint + object contract.
- Input transform is trivial Resize+ToTensor (no normalization/mean-std). Reproduced deterministically on CPU.

## Verdicts

| Item | Verdict |
|---|---|
| Reported pattern/category metrics independently reproduce | **YES (exact)** |
| Checkpoint ↔ config ↔ data consistent | YES |
| “Cyclone detection” headline capability | **NOT VERIFIABLE / NOT DEMONSTRATED** (synthetic labels, no negatives, no bbox truth) |
| Presence head evaluated | NO — never scored |
| Frontend consumes P2 | NO (no `/api/detect` anywhere in delivered backend) |

## p2_metrics.json

All of the above plus: checkpoint keys, label index maps, per-split file lists, presence-degeneracy checks, and the structural-pattern derivation-rule agreement (0.9925).