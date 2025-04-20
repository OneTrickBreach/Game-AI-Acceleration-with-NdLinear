import gymnasium as gym
import torch
import numpy as np
import os # Import os

from src.environments.env_wrapper import GymEnvWrapper
from src.agents.dqn_agent import DQNAgent
from src.training.train import train_agent, evaluate_agent
from src.utils.experiment_config import ACROBOT_CONFIG
from src.utils.model_utils import save_model, load_model # Import load_model

def train_acrobot():
    """Trains DQN agents on the Acrobot-v1 environment."""
    print("Training agents on Acrobot-v1 environment...")

    # Get experiment configuration
    config = ACROBOT_CONFIG

    # Create environment
    env = GymEnvWrapper(config['env_name'])

    # Get state and action sizes
    state_size = env.env.observation_space.shape[0]
    action_size = env.env.action_space.n

    # Define model paths
    baseline_model_path = "models/baseline/acrobot_dqn_baseline.pth"
    ndlinear_model_path = "models/ndlinear/acrobot_dqn_ndlinear.pth"

    # Initialize baseline agent
    print("\nInitializing baseline DQN agent...")
    baseline_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        use_ndlinear=False,
        seed=config['seed'] if 'seed' in config else 0
    )
    # Load baseline model if it exists
    if os.path.exists(baseline_model_path):
        baseline_agent.qnetwork_local = load_model(baseline_agent.qnetwork_local, baseline_model_path)
        print("Loaded existing baseline model.")

    # Train baseline agent
    print("Training baseline DQN agent...")
    hyperparameters = config['hyperparameters']
    baseline_scores = train_agent(baseline_agent, env, num_episodes=config['max_episodes'], save_filename=baseline_model_path, epsilon_start=hyperparameters['epsilon_start'] if 'epsilon_start' in hyperparameters else 1.0, epsilon_end=hyperparameters['epsilon_end'] if 'epsilon_end' in hyperparameters else 0.01, epsilon_decay=hyperparameters['epsilon_decay'] if 'epsilon_decay' in hyperparameters else 0.995)

    # Initialize NdLinear agent
    print("\nInitializing NdLinear DQN agent...")
    ndlinear_agent = DQNAgent(state_size, action_size, use_ndlinear=True, seed=config['seed'] if 'seed' in config else 0)
    # Load NdLinear model if it exists
    if os.path.exists(ndlinear_model_path):
        ndlinear_agent.qnetwork_local = load_model(ndlinear_agent.qnetwork_local, ndlinear_model_path)
        print("Loaded existing NdLinear model.")

    # Train NdLinear agent
    print("Training NdLinear DQN agent...")
    ndlinear_scores = train_agent(ndlinear_agent, env, num_episodes=config['max_episodes'], save_filename=ndlinear_model_path, epsilon_start=hyperparameters['epsilon_start'] if 'epsilon_start' in hyperparameters else 1.0, epsilon_end=hyperparameters['epsilon_end'] if 'epsilon_end' in hyperparameters else 0.01, epsilon_decay=hyperparameters['epsilon_decay'] if 'epsilon_decay' in hyperparameters else 0.995)

    env.close()

    # TODO: Plot results

    print("\nAcrobot-v1 training complete.")
    return baseline_scores, ndlinear_scores

if __name__ == '__main__':
    train_acrobot()