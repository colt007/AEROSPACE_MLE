import os
import sys
from pathlib import Path
# Ensure project root is on sys.path so `src` imports work when running the script directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow
import mlflow.pytorch
from src.data.dataloader import get_dataloaders
from src.models.model import CMAPSSModel

def main():
    config = {
        "input_size":24,
        "window_size": 30,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4, # Aggressive regularization for AdamW
        "epochs": 15,
        "hidden_size": 128,
        "train_split": 0.8
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"executing on {device}")
    parquet_path = "datasets/processed/output_parquet.parquet"
    train_loader,val_loader = get_dataloaders(parquet_path,window_size=config['window_size'],batch_size=config["batch_size"],train_split=config['train_split'])
    model = CMAPSSModel(input_size=config['input_size'],hidden_size=config['hidden_size']).to(device)
    optimizer = optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    
    mlflow.set_experiment('CMAPSS_BASELINE')
    with mlflow.start_run():
        mlflow.log_params(config)
        best_val_rmse = float('inf')
        os.makedirs('models',exist_ok=True)
        best_model_path = "models/first_model.pt"
        print("initializing training loop")
        for epoch in range(1,config['epochs']+1):
            model.train()
            train_loss = 0.0
            for batch_x,batch_y in train_loader:
                batch_x,batch_y = batch_x.to(device),batch_y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions,batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_rmse = np.sqrt(train_loss / len(train_loader.dataset))
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            
            val_rmse = np.sqrt(val_loss / len(val_loader.dataset))
            print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f}")
            mlflow.log_metrics({"train_rmse": train_rmse, "val_rmse": val_rmse}, step=epoch)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), best_model_path)
                print(f" -> Checkpoint Saved: Val RMSE dropped to {best_val_rmse:.2f}")
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        print(f"\nTraining Complete. Best Validation RMSE: {best_val_rmse:.2f} cycles.")

if __name__ == "__main__":
    main()
        
    
    