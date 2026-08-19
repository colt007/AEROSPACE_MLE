import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, NUM_FEATURES, WINDOW_SIZE
from src import CMAPSSModel

def test_model_can_overfit_single_batch():
    """Proves that gradients flow backward and the model can mathematically learn."""
    
    # 1. Setup
    model = CMAPSSModel(input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 2. Create Static Dummy Data
    batch_size = BATCH_SIZE
    window_size = WINDOW_SIZE
    num_features = NUM_FEATURES
    
    # We use a fixed seed so the test doesn't randomly fail due to bad initialization noise
    torch.manual_seed(42)
    X = torch.randn((batch_size, window_size, num_features))
    y = torch.rand((batch_size, 1)) * 100  # Fake RUL targets between 0 and 100
    
    # 3. Calculate Initial Loss
    model.train()
    initial_output = model(X)
    initial_loss = criterion(initial_output, y)
    
    # 4. Run 10 Training Steps on the exact same data
    for _ in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
    # 5. Calculate Final Loss
    final_output = model(X)
    final_loss = criterion(final_output, y)
    
    # 6. The Gradient Contract
    assert final_loss.item() < initial_loss.item(), (
        f"Gradients disconnected! Initial loss: {initial_loss.item()}, Final loss: {final_loss.item()}"
    )