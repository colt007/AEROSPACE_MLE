import pandas as pd
import numpy as np
import pytest



@pytest.fixture(scope='session')
def processed_df():
    return pd.read_parquet(r"C:\projects_data\ml_pipeline\data\output_parquet.parquet")

def test_engine_blocks_are_contiguous(processed_df):
    engine_sequence = processed_df["global_engine_id"]
    transitions = engine_sequence.ne(engine_sequence.shift())
    encountered = []
    for engine in engine_sequence[transitions]:
        if engine in encountered:
            pytest.fail(
                f"Engine {engine} appears in multiple disconnected blocks."
            )
        encountered.append(engine)
        
def test_time_cycle_is_strictly_consecutive(processed_df):
    grouped = processed_df.groupby("global_engine_id")
    for engine_id, group in grouped:
        diffs = group["time_cycle"].diff().dropna()
        assert (diffs == 1).all(), (
            f"Missing or non-consecutive cycles detected in {engine_id}"
        )
        
def test_rul_step_validity(processed_df):
    grouped = processed_df.groupby("global_engine_id")
    for engine_id, group in grouped:
        diffs = group["RUL"].diff().dropna()
        # RUL step must either be 0 (if capped during early healthy phase) or -1 (degrading)
        invalid_steps = diffs[~diffs.isin([0.0, -1.0])]
        assert invalid_steps.empty, (
            f"Invalid RUL step sizes in {engine_id}: {invalid_steps.values}"
        )
        
def test_rul_non_negative(processed_df):
    assert (processed_df["RUL"] >= 0).all(), (
        "Negative RUL values detected."
    )
    
def test_engine_final_rul_zero(processed_df):
    grouped = processed_df.groupby("global_engine_id")
    for engine_id, group in grouped:
        final_rul = group.iloc[-1]["RUL"]
        assert final_rul == 0, (
            f"{engine_id} ends with RUL={final_rul}"
        )
        
def test_dataset_not_empty(processed_df):
    assert len(processed_df) > 0, (
        "Processed dataset is empty."
    )
    
REQUIRED_COLUMNS = {
    "global_engine_id",
    "time_cycle",
    "RUL",
}

def test_required_columns_exist(processed_df):
    missing_columns = REQUIRED_COLUMNS - set(processed_df.columns)
    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )
    
def test_no_duplicate_rows(processed_df):
    duplicate_count = processed_df.duplicated().sum()
    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate rows."
    )
    
def test_unique_engine_cycle_pairs(processed_df):
    duplicates = processed_df.duplicated(
        subset=["global_engine_id", "time_cycle"]
    )
    duplicate_count = duplicates.sum()
    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicated engine-cycle pairs."
    )
        
def test_every_engine_has_data(processed_df):
    engine_sizes = processed_df.groupby("global_engine_id").size()
    assert (engine_sizes > 0).all(), (
        "One or more engines contain no observations."
    )

def test_no_null_or_infinite_values(processed_df):
    assert not processed_df.isna().any().any(), "NaN values detected in dataset."
    numeric_cols = processed_df.select_dtypes(include=["number"]).columns
    assert not np.isinf(processed_df[numeric_cols].values).any(), "Inf/-Inf values detected."

MIN_WINDOW_SIZE = 30

def test_engines_meet_minimum_sequence_length(processed_df):
    engine_lengths = processed_df.groupby("global_engine_id").size()
    short_engines = engine_lengths[engine_lengths < MIN_WINDOW_SIZE]
    assert short_engines.empty, (
        f"Engines found with fewer than {MIN_WINDOW_SIZE} cycles: {short_engines.to_dict()}"
    )


EXPECTED_SUBDATASETS = {"FD001", "FD002", "FD003", "FD004"}

def test_all_subdatasets_present(processed_df):
    # Extract prefix (e.g., "FD001" from "FD001_105")
    found_subdatasets = set(
        processed_df["global_engine_id"].str.split("_").str[0].unique()
    )
    
    assert found_subdatasets == EXPECTED_SUBDATASETS, (
        f"Expected subdatasets {EXPECTED_SUBDATASETS}, but found {found_subdatasets}"
    )
    