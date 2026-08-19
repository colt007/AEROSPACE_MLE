import torch
import pytest

from configs.config import BATCH_SIZE, HIDDEN_SIZE, NUM_FEATURES, WINDOW_SIZE
from src import CMAPSSModel


def test_model_forward_shape():
    """Proves the model can accept the correct matrix shape and output the correct prediction shape."""

    batch_size = BATCH_SIZE
    window_size = WINDOW_SIZE
    num_features = NUM_FEATURES
    hidden_size = HIDDEN_SIZE

    model = CMAPSSModel(input_size=num_features, hidden_size=hidden_size)

    dummy_input = torch.randn((batch_size, window_size, num_features), dtype=torch.float32)

    output = model(dummy_input)

    expected_shape = (batch_size, 1) 
    
    assert output.shape == expected_shape, (
        f"Matrix mismatch! Expected shape {expected_shape}, but model returned {output.shape}."
    )
    assert output.dtype == torch.float32, (
        "Precision mismatch! PyTorch models must output float32 for standard regression."
    )