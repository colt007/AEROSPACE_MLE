import asyncio
import json
import pickle
import mlflow
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import scipy.spatial.distance as dist


from configs.config import MLFLOW_TRACKING_URI, MODEL_URI, STATS_PATH, WINDOW_SIZE

import os
import mlflow

# In app.py before loading the model:
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(tracking_uri)


class SensorPayload(BaseModel):
    window_data: list[list[float]]


ROOT = Path(__file__).resolve().parents[2]
global_condition_matrices = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(STATS_PATH, "r") as f:
        app.state.stats = json.load(f)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    app.state.model = mlflow.pytorch.load_model(MODEL_URI, map_location="cpu")
    app.state.model.eval()
    global global_condition_matrices
    print("Loading Condition-Specific Mahalanobis Matrices...", flush=True)
    with open(ROOT / "src/serve/condition_matrices.pkl", "rb") as f:
        global_condition_matrices = pickle.load(f)
    yield

def preprocess_windows(window_data: list[list[float]], stats_dict: dict) -> torch.Tensor:
    processed_window = []
    
    for row in window_data:
        # Fix: Correct indices (0, 1, 2)
        first_op = int(abs(round(row[0], 0)))
        second_op = abs(round(row[1], 2))
        third_op = int(abs(round(row[2], 0)))
        
        op_combined = f"{first_op}_{second_op}_{third_op}"
        
        if op_combined not in stats_dict:
            raise ValueError(f"Unrecognized flight condition: {op_combined}")
            
        # Fix: Include the 3 operational settings so the tensor has 24 columns
        processed_row = [row[0], row[1], row[2]]
        
        for i in range(21):
            row_data = row[i + 3]
            sensor_idx = i + 1
            
            mean_sensor = stats_dict[op_combined][f"sensor_{sensor_idx}_mean"]
            std_sensor = stats_dict[op_combined][f"sensor_{sensor_idx}_std"]
            
            if std_sensor == 0.0:
                if abs(row_data-mean_sensor)>1e-4:
                    raise HTTPException(status_code=400,detail=f"Hardware Failure: Sensor{sensor_idx} deviating from constant baseline")
                    
                else:
                    norm_val = 0.0
            else:
                norm_val = (row_data - mean_sensor) / std_sensor
                
            processed_row.append(norm_val)
            
        processed_window.append(processed_row)
        
    tensor_window = torch.tensor(processed_window, dtype=torch.float32)
    tensor_window = tensor_window.unsqueeze(0)
    
    return tensor_window


def run_model_inference(model, tensor_values: torch.Tensor) -> torch.Tensor:
    """Run PyTorch inference in a worker thread without tracking gradients."""
    with torch.inference_mode():
        return model(tensor_values)

# Fix: Correct FastAPI initialization
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Vite's local ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(payload: SensorPayload):
    if len(payload.window_data) != WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Window must contain exactly {WINDOW_SIZE} time steps.")
        
    try:
        tensor_values = preprocess_windows(payload.window_data, app.state.stats)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    device = next(app.state.model.parameters()).device
    tensor_values = tensor_values.to(device)
  
    raw_window_data = payload.window_data
    current_raw_row = raw_window_data[-1]
    first_op = int(abs(round(current_raw_row[0], 0)))
    second_op = abs(round(current_raw_row[1], 2))
    third_op = int(abs(round(current_raw_row[2], 0)))
    current_condition_key = f"{first_op}_{second_op}_{third_op}"

    normalized_array = tensor_values.squeeze(0).cpu().numpy()
    current_normalized_sensors = normalized_array[-1][3:]

    try:
        condition_matrices = global_condition_matrices[current_condition_key]
        mu = condition_matrices["mu"]
        inv_cov = condition_matrices["inv_cov"]
        distance = dist.mahalanobis(current_normalized_sensors, mu, inv_cov)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"No safety matrices found for condition: {current_condition_key}",
        )

    print(
        f"\n[SAFETY VALVE] Condition: {current_condition_key} | "
        f"Distance: {distance:.2f}",
        flush=True,
    )
    print(f"[SAFETY VALVE] Statistical Limit: ~7.15\n", flush=True)
    if distance > 7.15:
        raise HTTPException(status_code=400, detail=f"OOD Data Detected. Distance {distance:.2f} exceeds threshold.")
    predictions = await asyncio.to_thread(
        run_model_inference,
        app.state.model,
        tensor_values,
    )
    return {"predicted_rul": float(predictions.item())}