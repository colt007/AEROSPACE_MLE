import numpy as np
import pandas as pd
import pytest

from configs.config import WINDOW_SIZE
from src import CMAPSSDataset


@pytest.fixture
def mock_engine_df():
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


def test_sequence_leakage(mock_engine_df):
    dataset = CMAPSSDataset(dataframe=mock_engine_df, window_size=WINDOW_SIZE)
    assert len(dataset) == 32, f"expected 32 windows got {len(dataset)}"

    features, _ = dataset[21]
    assert np.all(features[:, 0] == 2.0), "boundary breached between engines"
    