# Model Strength Summary

Signed, honest ranking of the components by *demonstrated* and *independently verified* strength. “Strength” here = defensible predictive value on held-out data, not just reported numbers.

## Verified FIT of each delivered model

| Component | What it is | Verified reproduced | Honest capability |
|---|---|---|---|
| P2 detection `model_weights.pt` | MobileNetV3-small, 3 heads | pattern acc 0.714 / category acc 0.333 on 21 frames | **Weak single-frame classifier over synthetic labels.** No real detection, no bbox, no negatives. 21-frame test = statistically meaningless (CI ±20%+). |
| P3 image `image_only_model.pt` | ResNet18, 7-class + wind regressor | acc 38.1%, macro-F1 0.21, wind MAE/RMSE 109.8/118.4 | **Weak** on a 21-frame test with leak-prone same-storm train frames; wind regression error ≈ 45% of range. |
| P3 tabular `.pkl` | LightGBM multi-output | **NOT_RUN** (env) | reported acc 47% / wind MAE 18.8 but on a *different* population; self-comparison vs image invalid. Treated as UNVERIFIED. |
| P4 EXP005 champion | GRU(96,2)+Huber | test 6/12/24h track 91.4/119.8/188.2 km; wind 6.7/9.5/16.1 MAE | **Only component with clean methodology and fully reproduced scores. But track accuracy is BELOW the movement-vector baseline at every horizon.** Wind forecasting is the truthful bright spot (small MAE vs magnitude), yet track is the headline. |
| P4 baselines | persistence / movement-vector | reproduced to machine precision | persistence 64.7/123.9/227.3; movement-vec 38.1/80.5/180.7 — **superior to the champion on track.** |
| P5 dashboard | React SPA | mock-only (consistent UI) | Demonstrative; no model data behind it yet. |

## Ranking (most to least defensible claim)

1. **P4 wind-intensity forecasting** (champion): reproduced; good relative to magnitude; does not beat track baselines but is the only verified real signal.
2. **P4 pipeline hygiene**: causality, normalization, selection, test-once — exemplary and verified.
3. **P2/P3 image metrics**: reproduced exactly, but test design (21 frames, synthetic labels, leak-prone near-dup storms) makes them unimpressive evidence.
4. **Track forecasting**: weakest among the *claims* that matter — trumped by a 30-line baseline.

## Direct quotes not supported by evidence (do not carry forward)

- “Multi-source classification beats image-only by +8.9% accuracy / −91 km/h MAE” — different test populations.
- “Detection accuracy ≈ 71%/33%” as detection — it is abled as pattern/category-only on synthetic labels; presence/bbox not scored.
- Test scores imply “future prediction” — random (not chronological) split; scores are on static held-out storms during the same era.

## What IS productizable today without further training

- P4 wind forecast + track output with explicit baseline-context caveats (i.e., present movement-vector as the reference, and be honest the NN does not beat it on track).
- Dashboard as an *operational mock* with phase6 `/forecast` wired through a small adapter (honest labelling “demonstration”).