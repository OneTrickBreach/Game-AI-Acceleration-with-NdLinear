import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_reward_comparison(baseline_rewards, ndlinear_rewards, title='Reward Comparison'):
    """Plots the reward comparison between baseline and NdLinear agents."""
    print(f"Generating {title} plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(baseline_rewards)), baseline_rewards, label='Baseline DQN')
    plt.plot(np.arange(len(ndlinear_rewards)), ndlinear_rewards, label='NdLinear DQN')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_inference_time(baseline_times, ndlinear_times, title='Inference Time Comparison', save_path=None):
    """Plots the inference time comparison."""
    print(f"Generating {title} plot...")
    labels = ['Baseline DQN', 'NdLinear DQN']
    times = [baseline_times, ndlinear_times]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('Average Time per Step (seconds)')
    plt.title(title)
    plt.grid(axis='y')
    # plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_memory_usage(baseline_memory, ndlinear_memory, title='Memory Usage Comparison', save_path=None):
    """Plots the memory usage comparison."""
    print(f"Generating {title} plot...")
    labels = ['Baseline DQN', 'NdLinear DQN']
    memory = [baseline_memory, ndlinear_memory]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, memory, color=['blue', 'orange'])
    plt.ylabel('Memory Usage (MB)')
    plt.title(title)
    plt.grid(axis='y')
    # plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_size(baseline_size, ndlinear_size, title='Model Size Comparison', save_path=None):
    """Plots the model size comparison."""
    print(f"Generating {title} plot...")
    labels = ['Baseline DQN', 'NdLinear DQN']
    size = [baseline_size, ndlinear_size]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, size, color=['blue', 'orange'])
    plt.ylabel('Model Size (MB)')
    plt.title(title)
    plt.grid(axis='y')
    # plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()