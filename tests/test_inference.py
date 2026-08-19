import torch
from src import CMAPSSModel

def test_inference_determinism():
    model = CMAPSSModel(input_size=24, hidden_size=128)
    X = torch.randn((1, 30, 24))
    model.eval()
    with torch.no_grad():
        output_1 = model(X)
        output_2 = model(X)
    assert torch.equal(output_1, output_2), (
        "Inference is non-deterministic! Layers like Dropout are still active."
    )

def test_inference_mode_saves_memory():
    """Proves that inference disables the gradient tracking graph."""
    model = CMAPSSModel(input_size=24, hidden_size=128)
    X = torch.randn((1, 30, 24))
    
    with torch.no_grad():
        output = model(X)
        
    assert output.requires_grad is False, (
        "Memory leak detected! Gradient tracking is still active during inference."
    )