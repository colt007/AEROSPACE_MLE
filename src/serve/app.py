import json
import mlflow
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

class SensorPayload(BaseModel):
    window_data: list[list[float]]
    


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Stats
    dict_path = Path("C:/MLE/aerospace_mle/datasets/processed/output_json.json")
    with open(dict_path, "r") as f:
        app.state.stats = json.load(f)
        
    # Load Model from MLflow
    run_id = "34a55fbcd4b6426c98e9d24e673c0d35"
    model_uri = f"runs:/{run_id}/model"
    app.state.model = mlflow.pytorch.load_model(model_uri)
    
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
                norm_val = 0.0
            else:
                norm_val = (row_data - mean_sensor) / std_sensor
                
            processed_row.append(norm_val)
            
        processed_window.append(processed_row)
        
    tensor_window = torch.tensor(processed_window, dtype=torch.float32)
    tensor_window = tensor_window.unsqueeze(0)
    
    return tensor_window

# Fix: Correct FastAPI initialization
app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(payload: SensorPayload):
    # Fix: Pydantic payload unpacking and proper HTTPException handling
    if len(payload.window_data) != 30:
        raise HTTPException(status_code=400, detail="Window must contain exactly 30 time steps.")
        
    try:
        tensor_values = preprocess_windows(payload.window_data, app.state.stats)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    device = next(app.state.model.parameters()).device
    tensor_values = tensor_values.to(device)
        
    with torch.no_grad():
        predictions = app.state.model(tensor_values)
        
    # Fix: Return standard Python dict, FastAPI converts it to JSON automatically
    return {"predicted_rul": float(predictions.item())}