import gymnasium as gym
import torch
import argparse
import os

from src.environments.env_wrapper import GymEnvWrapper
from src.agents.dqn_agent import DQNAgent
from src.utils.model_utils import load_model

VALID_ENVS = ["CartPole-v1", "MountainCar-v0", "Pendulum-v1", "Acrobot-v1"]

def main():
    parser = argparse.ArgumentParser(description="Demonstrate a trained agent in a game environment.")
    parser.add_argument("--env", type=str, required=True, help="The name of the environment (e.g., CartPole-v1, MountainCar-v0)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file)")
    parser.add_argument("--render", action="store_true", help="Enable rendering of the environment")
    args = parser.parse_args()

    # Validate the environment
    if args.env not in VALID_ENVS:
        print(f"Error: Invalid environment '{args.env}'. Valid environments are: {VALID_ENVS}")
        return

    # Load the environment
    render_mode = "human" if args.render else None
    env = GymEnvWrapper(args.env, render_mode=render_mode)
    state_size = env.env.observation_space.shape[0]
    action_size = env.env.action_space.n

    # Determine if the model is NdLinear
    use_ndlinear = "ndlinear" in args.model_path.lower()

    # Load the agent
    try:
        agent = DQNAgent(state_size, action_size, use_ndlinear=use_ndlinear)
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at '{args.model_path}'")
            return
        agent.qnetwork_local = load_model(agent.qnetwork_local, args.model_path)
        agent.qnetwork_local.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run the agent
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state, epsilon=0.0) # Greedy action selection
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        # Rendering is now handled by the environment wrapper based on render_mode
        # if args.render:
        #     env.env.render() 

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()