"""P1 data-pipeline assertions (facts recorded in p1_metrics.json)."""

import io

import pytest
import zipfile


def test_master_counts_and_geometry(p1):
    assert p1["master_rows"] == 5481
    assert p1["master_cyclones"] == 151
    assert p1["master_dup_rows_exact"] == 0
    assert p1["master_dup_cyc_tstamp"] == 0
    assert p1["master_rows_minus_split_rows"] == 0


def test_split_disjointness(p1):
    inter = p1["split_cyclone_intersections"]
    assert inter["train&val"] == 0
    assert inter["train&test"] == 0
    assert inter["val&test"] == 0
    assert p1["split_cyclone_union_matches_master"] is True
    assert p1["split_cyclones"] == {"train": 105, "val": 22, "test": 24}


def test_era5_join_and_known_missingness(p1):
    assert p1["era5j_rows"] == 5481
    assert p1["sst_missing"] == 1538
    assert p1["phys_out_of_range_count"]["lat"] == 0
    assert p1["phys_out_of_range_count"]["lon"] == 0
    assert p1["phys_out_of_range_count"]["wind_kmh"] == 0
    assert p1["phys_out_of_range_count"]["sst_ok"] == 0


def test_pipeline_counts_match_qa(p1):
    assert p1["clean_rows"] == 18168
    assert p1["clean_cyclones"] == 471
    assert p1["cls_test_rows"] == 651
    assert p1["cls_train_rows"] == 3039
    assert p1["cls_val_rows"] == 518


def test_detection_labels_are_synthetic(p1):
    assert p1["det_all_detected_true"] is True
    assert p1["det_mock_bbox_unique"] == 1
    assert p1["det_all_image_paths_resolve"] is True
    assert p1["det_train_same_files_as_kaggle_train"] is True
    assert p1["det_val_same_files_as_kaggle_val"] is True
    assert p1["det_test_same_files_as_kaggle_test"] is True


def test_forecast_sequence_shapes_no_nan(p1):
    assert p1["seq_train_X_shape"] == [2275, 5, 7]
    assert p1["seq_val_X_shape"] == [378, 5, 7]
    assert p1["seq_test_X_shape"] == [423, 5, 7]
    assert p1["seq_train_nan_X"] == 0 and p1["seq_train_nan_Y"] == 0
    assert p1["seq_test_nan_X"] == 0 and p1["seq_test_nan_Y"] == 0


def test_qa_completeness_wording_inconsistent(p1):
    # The README-level "Completeness: 100.0%" is contradicted by the audit's own
    # documented sst missingness. The report must not present 100% completeness.
    assert p1["sst_missing"] > 0


def test_image_split_leak_flag(p1):
    # same-storm frames straddle splits (near-duplicate image leakage)
    ov = p1["img_split_base_storm_overlap"]
    assert ov["test+train"] >= 0
    assert (ov["test+train"] + ov["train+val"] + ov["test+val"]) >= 10
    assert p1["kaggle_byte_duplicates_any"] is False


def test_master_class_consistency(p1):
    assert p1["master_wind2cat_agree"] == 1.0
    assert p1["category_counts"]["Depression"] == 1665
    assert p1["category_counts"]["Super Cyclonic Storm"] == 25


def test_spreadsheet_header_stability(p1):
    assert p1["insat_sheet_cols"] == ["img_name", "label"]
    assert int(p1["insat_sheet_rows"]) == 136