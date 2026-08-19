import json
import torch
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

# 1. Define Paths (Adjust these if your directories are different)
raw_test_path = Path("C:/projects_data/data/raw/test_FD001.txt")
raw_rul_path = Path("C:/projects_data/data/raw/RUL_FD001.txt")
stats_path = Path("C:/MLE/aerospace_mle/datasets/processed/output_json.json")

# 2. Load the Model from MLflow and move to GPU if available
print("Loading architecture and weights...")
run_id = "34a55fbcd4b6426c98e9d24e673c0d35"  # Your successful run ID
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri)
device = next(model.parameters()).device
model.eval()

# 3. Load the Normalization Statistics
with open(stats_path, "r") as f:
    stats_dict = json.load(f)

# 4. Load the Ground Truth RULs
# RUL_FD001.txt has one number per line, corresponding to Engine 1, Engine 2, etc.
print("Loading Ground Truth labels...")
true_ruls = pd.read_csv(raw_rul_path, header=None, names=['RUL'])
true_rul_array = true_ruls['RUL'].values

# 5. Load the Raw Test Data
print("Loading and processing raw test sensor data...")
columns = ['unit_nr', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
test_df = pd.read_csv(raw_test_path, sep=r'\s+', header=None, names=columns)

predictions = []
actuals = []

# 6. Process exactly the last 30 cycles for each engine
engine_ids = test_df['unit_nr'].unique()

for engine_id in engine_ids:
    engine_data = test_df[test_df['unit_nr'] == engine_id]
    
    # We only want the final 30 flights to predict the final RUL
    if len(engine_data) < 30:
        continue # Skip engines that don't have enough history for our window size
        
    window_df = engine_data.tail(30).drop(columns=['unit_nr', 'time_cycles'])
    window_data = window_df.values.tolist()
    
    # 7. Apply the exact API Normalization Logic (The Safety Valve)
    processed_window = []
    for row in window_data:
        op1 = int(abs(round(row[0], 0)))
        op2 = abs(round(row[1], 2))
        op3 = int(abs(round(row[2], 0)))
        op_combined = f"{op1}_{op2}_{op3}"
        
        processed_row = [row[0], row[1], row[2]] # Keep raw ops
        
        for i in range(21):
            raw_val = row[i + 3]
            sensor_idx = i + 1
            mean_sensor = stats_dict[op_combined][f"sensor_{sensor_idx}_mean"]
            std_sensor = stats_dict[op_combined][f"sensor_{sensor_idx}_std"]
            
            norm_val = 0.0 if std_sensor == 0.0 else (raw_val - mean_sensor) / std_sensor
            processed_row.append(norm_val)
            
        processed_window.append(processed_row)
        
    # Convert to Tensor, map to device, and predict
    tensor_window = torch.tensor(processed_window, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_rul = model(tensor_window).item()
        
    predictions.append(pred_rul)
    # engine_id is 1-indexed, array is 0-indexed
    actuals.append(true_rul_array[engine_id - 1])

# 8. Calculate Final Mathematical Score
rmse = np.sqrt(mean_squared_error(actuals, predictions))
print("-" * 40)
print(f"EVALUATION COMPLETE")
print(f"Total Engines Evaluated: {len(predictions)}")
print(f"FINAL TEST RMSE: {rmse:.2f} cycles")
print("-" * 40)