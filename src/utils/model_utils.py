import torch
import os

def save_model(model, filename):
    """Saves the state dictionary of a PyTorch model."""
    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename):
    """Loads a saved state dictionary into a PyTorch model."""
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
    else:
        print(f"Model file not found at {filename}")
    return model