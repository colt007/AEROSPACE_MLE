import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets/processed/output_parquet.parquet"
OUTPUT_PATH = ROOT / "src/serve/condition_matrices.pkl"

OPERATIONAL_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]


def condition_key(values) -> str:
    first_op = int(abs(round(values.iloc[0], 0)))
    second_op = abs(round(values.iloc[1], 2))
    third_op = int(abs(round(values.iloc[2], 0)))
    return f"{first_op}_{second_op}_{third_op}"


def safety_valve(
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, dict[str, np.ndarray]]:
    data = pd.read_parquet(data_path)
    required_columns = OPERATIONAL_COLUMNS + SENSOR_COLUMNS
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data = data.copy()
    data["condition_key"] = data[OPERATIONAL_COLUMNS].apply(condition_key, axis=1)
    condition_matrices = {}

    for key, group in data.groupby("condition_key", sort=True):
        sensor_data = group[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        covariance = np.cov(sensor_data, rowvar=False)
        condition_matrices[key] = {
            "mu": np.mean(sensor_data, axis=0),
            "inv_cov": np.linalg.pinv(covariance),
        }
        print(f"Processing condition {key}: {len(group):,} rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as matrix_file:
        pickle.dump(condition_matrices, matrix_file)

    print(f"Saved {len(condition_matrices)} condition matrices to {output_path}")
    return condition_matrices


if __name__ == "__main__":
    safety_valve()