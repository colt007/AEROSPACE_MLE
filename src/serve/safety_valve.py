import pandas as pd
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def safety_valve():
    data = pd.read_parquet(ROOT / "datasets/processed/output_parquet.parquet")
    feature_cols = [cols for cols in data.columns if cols not in ['time_cycle','global_engine_id','RUL']]
    data_values = data[feature_cols].values
    mean_vector = np.mean(data_values,axis=0)
    cov_matrix = np.cov(data_values,rowvar=False)
    inverse_conv_matrix = np.linalg.pinv(cov_matrix)
    np.save(ROOT/"src/serve/mu.npy", mean_vector)
    np.save(ROOT/"src/serve/inv_cov.npy", inverse_conv_matrix)
    
if __name__=="__main__":
    safety_valve()
    