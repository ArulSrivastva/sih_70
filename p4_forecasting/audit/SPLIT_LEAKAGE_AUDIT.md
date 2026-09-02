# SPLIT LEAKAGE AUDIT (P4 PHASE 1)

## npz-level cyclone overlap (sequences)

| pair | overlapping cyclone IDs | count |
|---|---|---|
| train_vs_val | - | 0 |
| train_vs_test | - | 0 |
| val_vs_test | - | 0 |

## Master-level cyclone lists (P1 split files)

| check | value |
|---|---|
| train_vs_val | 0 |
| train_vs_test | 0 |
| val_vs_test | 0 |
| npz_train_within_master_train | 67 |

## Duplicate detection (npz level)

| split | dup (cyclone_id, t_zero) | dup metadata rows | dup X rows | dup Y rows | dup X+Y |
|---|---|---|---|---|---|
| train | 0 | 0 | 0 | 1 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

Conclusion: no duplicate sequences, no duplicate pairs, splits fully disjoint.
