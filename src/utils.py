import torch

def get_device():
    """Check and return available device (GPU/CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_model_stats(model):
    """Print model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
