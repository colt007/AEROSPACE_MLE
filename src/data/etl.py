import pandas as pd


def normalize_sensors(data):
    """Apply condition-based Z-score normalization to sensor data."""
    exclude_cols = [
        "time_cycle",
        "global_engine_id",
        "RUL",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        "Condition_ID",
    ]
    sensor_columns = [col for col in data.columns if col not in exclude_cols]

    lookup_table = data.groupby("Condition_ID")[sensor_columns].agg(["mean", "std"])
    lookup_table.columns = [f"{col[0]}_{col[1]}" for col in lookup_table.columns.values]
    lookup_table = lookup_table.reset_index()

    data = pd.merge(data, lookup_table, on="Condition_ID", how="left")

    epsilon = 1e-8
    for col in sensor_columns:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        data[col] = (data[col] - data[mean_col]) / (data[std_col] + epsilon)

    columns_to_drop = [f"{col}_mean" for col in sensor_columns] + [f"{col}_std" for col in sensor_columns]
    data = data.drop(columns=columns_to_drop)

    artifact_dict = lookup_table.set_index("Condition_ID").to_dict(orient="index")
    return data, artifact_dict
