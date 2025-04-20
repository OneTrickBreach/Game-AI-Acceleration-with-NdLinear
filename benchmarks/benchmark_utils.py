import time
import torch
import numpy as np
import psutil
import os

def measure_inference_time(agent, env, num_steps=1000):
    """Measures the average inference time of a model in an environment."""
    print(f"Measuring inference time for {type(agent).__name__}...")
    state = env.reset()
    start_time = time.time()
    for _ in range(num_steps):
        # action = model.select_action(state, epsilon=0.0) # Use greedy policy
        # state, _, done, _ = env.step(action)
        action = agent.select_action(state)
        # Try stepping with the raw action; if the env expects a vector, wrap it
        try:
            state, _, done, _ = env.step(action)
        except (IndexError, TypeError):
            import numpy as np
            state, _, done, _ = env.step(np.array([action], dtype=np.float32))
        if done:
            state = env.reset()
    end_time = time.time()
    avg_time_per_step = (end_time - start_time) / num_steps
    print(f"Average inference time per step: {avg_time_per_step:.6f} seconds")
    return avg_time_per_step

def measure_memory_usage(model):
    """Measures the memory usage of a model."""
    print(f"Measuring memory usage for {type(model).__name__}...")
    # This is a basic approach and might not capture all memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / (1024 * 1024) # Resident Set Size in MB
    print(f"Memory usage: {memory_mb:.2f} MB")
    return memory_mb

def count_parameters(model):
    """Counts the total number of parameters in a model."""
    print(f"Counting parameters for {type(model).__name__}...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    return total_params

def measure_model_size(model, filename="temp_model.pth"):
    """Measures the size of a model in MB by saving it to a temporary file."""
    print(f"Measuring model size for {type(model).__name__}...")
    torch.save(model.state_dict(), filename)
    model_size_mb = os.path.getsize(filename) / (1024 * 1024)
    os.remove(filename)
    print(f"Model size: {model_size_mb:.2f} MB")
    return model_size_mb