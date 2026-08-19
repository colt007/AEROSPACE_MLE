import numpy as np
import pandas as pd
import pytest
from src import normalize_sensors

@pytest.fixture(scope = 'session')
def mock_df():
    return pd.DataFrame({
        "Condition_ID": [
            "Cond_A", "Cond_A", "Cond_A", "Cond_A", "Cond_A",
            "Cond_B", "Cond_B", "Cond_B", "Cond_B", "Cond_B"
        ],
        # Condition A runs hot (mean ~500), Condition B runs cold (mean ~10)
        "sensor_1": [500.0, 510.0, 490.0, 505.0, 495.0, 
                     10.0, 12.0, 8.0, 11.0, 9.0],
        "sensor_2": [100.0, 102.0, 98.0, 101.0, 99.0, 
                     2.0, 2.5, 1.5, 2.2, 1.8]
    })

def test_normalization(mock_df):
    normalized_df, artifact_dict = normalize_sensors(mock_df)
    grouped = normalized_df.groupby("Condition_ID")
    
    # 3. Assert the Physics Contract for every condition independently
    for condition, group in grouped:
        
        # Sensor 1 Check
        s1_mean = group["sensor_1"].mean()
        s1_std = group["sensor_1"].std()
        
        assert np.isclose(s1_mean, 0.0, atol=1e-5), (
            f"Failed in {condition}: sensor_1 mean is {s1_mean}, expected 0.0"
        )
        assert np.isclose(s1_std, 1.0, atol=1e-5), (
            f"Failed in {condition}: sensor_1 std is {s1_std}, expected 1.0"
        )
        
        # Sensor 2 Check
        s2_mean = group["sensor_2"].mean()
        s2_std = group["sensor_2"].std()
        
        assert np.isclose(s2_mean, 0.0, atol=1e-5), (
            f"Failed in {condition}: sensor_2 mean is {s2_mean}, expected 0.0"
        )
        assert np.isclose(s2_std, 1.0, atol=1e-5), (
            f"Failed in {condition}: sensor_2 std is {s2_std}, expected 1.0"
        )
    
    
    