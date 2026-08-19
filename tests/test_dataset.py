import numpy as np
import pandas as pd
import pytest

from configs.config import WINDOW_SIZE
from src import CMAPSSDataset

@pytest.fixture(scope='session')
def mock_df():
    """Generates a small in-memory engine DataFrame for fast testing."""
    cycles = 50
    engine_A = pd.DataFrame({
        "global_engine_id": ["FD001_1"] * cycles,
        "time_cycle": np.arange(1, cycles + 1),
        "sensor_1": np.ones(cycles).astype(np.float32),
        "sensor_2": np.random.randn(cycles).astype(np.float32),
        "RUL": np.arange(cycles - 1, -1, -1, dtype=np.float32)
    })
    b_cycles = 40
    engine_B = pd.DataFrame({
        "global_engine_id": ["FD001_2"] * b_cycles,
        "time_cycle": np.arange(1, b_cycles + 1),
        "sensor_1": np.full(b_cycles, 2.0, dtype=np.float32),
        "sensor_2": np.random.randn(b_cycles).astype(np.float32),
        "RUL": np.arange(b_cycles - 1, -1, -1, dtype=np.float32)
    })
    return pd.concat([engine_A, engine_B], ignore_index=True)

def test_windowing_no_windows(mock_df):
    dataset = CMAPSSDataset(dataframe=mock_df, window_size=WINDOW_SIZE)
    assert len(dataset) ==32, f"expected 32 windows, but received {len(dataset)} "
    
def test_windowing_boundary_preserved(mock_df):
    dataset = CMAPSSDataset(dataframe=mock_df, window_size=WINDOW_SIZE)
    features,targets = dataset[21]
    assert np.all(features[:,0]==2.0),"boundary breached engines combined"
    
def test_windowing_dtype(mock_df):
    dataset = CMAPSSDataset(dataframe=mock_df, window_size=WINDOW_SIZE)
    features,targets = dataset[0]
    
    assert features.dtype == np.float32 ,f"expected float32 received {features.dtype}"
    assert targets.dtype == np.float32, f"expected float32 received {targets.dtype}"
    
def test_window_shape(mock_df):
    dataset = CMAPSSDataset(dataframe=mock_df, window_size=WINDOW_SIZE)
    features,targets = dataset[0]
    assert features.shape == (WINDOW_SIZE,2), f"expected ({WINDOW_SIZE},2) but received {features.shape} shape for features"
    assert np.isscalar(targets) , "expected scalar for targets"
    
def test_target_alignment_first_window(mock_df):
    dataset = CMAPSSDataset(
        dataframe=mock_df,
        window_size=WINDOW_SIZE
    )

    _, target = dataset[0]

    assert target == np.float32(20), (
        f"Expected target 20, received {target}"
    )
    
def test_target_alignment_last_window(mock_df):
    dataset = CMAPSSDataset(
        dataframe=mock_df,
        window_size=WINDOW_SIZE
    )

    _, target = dataset[20]

    assert target == np.float32(0), (
        f"Expected target 0, received {target}"
    )

def test_window_continuation(mock_df):
    dataset = CMAPSSDataset(
        dataframe=mock_df,
        window_size=WINDOW_SIZE
    )
    features0,targets = dataset[0]
    features1,targets  = dataset[1]
    assert np.array_equal(
    features0[1:],
    features1[:-1]
    ), "Sliding window overlap violated."

def test_smaller_window(mock_df):
    cycles = 20
    temp_df = pd.DataFrame({
        "global_engine_id": ["FD001_1"] * cycles,
        "time_cycle": np.arange(1, cycles + 1),
        "sensor_1": np.ones(cycles).astype(np.float32),
        "sensor_2": np.random.randn(cycles).astype(np.float32),
        "RUL": np.arange(cycles - 1, -1, -1, dtype=np.float32)
    })
    
    dataset = CMAPSSDataset(dataframe=temp_df, window_size=WINDOW_SIZE)
    assert (len(dataset)==0) ,f"expected zero for window size but got {len(dataset)}"
    
    

    
    
    
    
