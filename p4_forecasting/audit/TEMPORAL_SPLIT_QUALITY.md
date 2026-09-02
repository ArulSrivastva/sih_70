# TEMPORAL SPLIT QUALITY (P4 PHASE 1)

Question: is the P1 npz split chronological, cyclone-grouped, or random?

## Split: train

- cyclones: 67
- min t_zero: 2013-05-10 18:00:00
- max t_zero: 2025-11-29 06:00:00
- year span : 2013 .. 2025
- years present: [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

## Split: val

- cyclones: 14
- min t_zero: 2013-11-07 00:00:00
- max t_zero: 2025-10-28 18:00:00
- year span : 2013 .. 2025
- years present: [2013, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

## Split: test

- cyclones: 16
- min t_zero: 2013-05-30 03:00:00
- max t_zero: 2025-12-01 18:00:00
- year span : 2013 .. 2025
- years present: [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2024, 2025]

Result: every split contains 2013..2025 cyclones, so the P1 npz split is a RANDOM (non-chronological), cyclone-grouped 70/15/15 split. Cyclone-level grouping is correct (no split leakage).

Primary canonical split keeps P1's random grouping. A secondary CHRONOLOGICAL split is provided in p4_forecasting/canonical_chrono/ to support time-ordered validation.
