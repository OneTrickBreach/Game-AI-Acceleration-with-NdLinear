import argparse
import gymnasium as gym
import torch
import numpy as np
import random

from src.environments.env_wrapper import GymEnvWrapper
from src.agents.dqn_agent import DQNAgent  # still used for CartPole, etc.
from stable_baselines3 import PPO           # for Pendulum
from src.training.train import train_agent, evaluate_agent
from src.training.cartpole_training import train_cartpole
from src.training.mountaincar_training import train_mountaincar
from src.training.pendulum_training import train_pendulum
from src.training.acrobot_training import train_acrobot
from src.utils.experiment_config import CARTPOLE_CONFIG, MOUNTAINCAR_CONFIG, PENDULUM_CONFIG, ACROBOT_CONFIG
from benchmarks.run_benchmarks import run_all_benchmarks
from visualization.performance_plots import plot_reward_comparison, plot_inference_time, plot_memory_usage, plot_model_size
from visualization.record_agent import record_episode
from src.utils.model_utils import load_model # Import load_model
from src.training.pendulum_training import train_pendulum
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description='Game AI Acceleration with NdLinear')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'benchmark', 'record'], help='Mode: train, evaluate, benchmark, record')
    parser.add_argument('--model', type=str, default='both', choices=['standard', 'ndlinear', 'both'], help='Model: standard, ndlinear, both')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training or evaluation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # Add more arguments as needed (e.g., hyperparameters)

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Get environment config
    if args.env == 'CartPole-v1':
        config = CARTPOLE_CONFIG
        train_env_func = train_cartpole
        env_name_short = 'cartpole'
    elif args.env == 'MountainCar-v0':
        config = MOUNTAINCAR_CONFIG
        train_env_func = train_mountaincar
        env_name_short = 'mountaincar'
    elif args.env == 'Pendulum-v1':
        config = PENDULUM_CONFIG
        train_env_func = train_pendulum
        env_name_short = 'pendulum'
    elif args.env == 'Acrobot-v1':
        config = ACROBOT_CONFIG
        train_env_func = train_acrobot
        env_name_short = 'acrobot'
    else:
        raise ValueError(f"Environment {args.env} not supported.")

    # Create environment for evaluation/benchmarking/recording
    if args.mode != 'train':
         env = GymEnvWrapper(config['env_name'])
         state_size = env.env.observation_space.shape[0]
         # Determine action size based on environment type (discrete/continuous)
         if isinstance(env.env.action_space, gym.spaces.Discrete):
             action_size = env.env.action_space.n
         else:
             action_size = env.env.action_space.shape[0]


    # Run in specified mode
    if args.mode == 'train':
        train_env_func() # Call environment-specific training function

    elif args.mode == 'evaluate':
        if args.env == "Pendulum-v1":
            print(f"Loading PPO policy from {PENDULUM_CONFIG['model_path']} …")
            eval_env = gym.make(PENDULUM_CONFIG['env_name'], render_mode="human")
            model = PPO.load(PENDULUM_CONFIG['model_path'], env=eval_env)
            obs, _ = eval_env.reset()
            for _ in range(args.num_episodes):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                # eval_env.render()
                if terminated or truncated:
                    obs, _ = eval_env.reset()
            eval_env.close()
            return

        baseline_agent = None
        ndlinear_agent = None

        if args.model == 'standard' or args.model == 'both':
            # Load baseline model
            baseline_agent = DQNAgent(state_size, action_size, use_ndlinear=False, seed=args.seed)
            model_path = f"models/baseline/{env_name_short}_dqn_baseline.pth"
            baseline_agent.qnetwork_local = load_model(baseline_agent.qnetwork_local, model_path)
            print("Evaluating baseline agent...")
            evaluate_agent(baseline_agent, env, num_episodes=args.num_episodes)
        if args.model == 'ndlinear' or args.model == 'both':
            # Load NdLinear model
            ndlinear_agent = DQNAgent(state_size, action_size, use_ndlinear=True, seed=args.seed)
            model_path = f"models/ndlinear/{env_name_short}_dqn_ndlinear.pth"
            ndlinear_agent.qnetwork_local = load_model(ndlinear_agent.qnetwork_local, model_path)
            print("Evaluating NdLinear agent...")
            evaluate_agent(ndlinear_agent, env, num_episodes=args.num_episodes)

    elif args.mode == 'benchmark':
        baseline_agent = None
        ndlinear_agent = None

        if args.model == 'both':
             # Load both models
            baseline_agent = DQNAgent(state_size, action_size, use_ndlinear=False, seed=args.seed)
            baseline_model_path = f"models/baseline/{env_name_short}_dqn_baseline.pth"
            baseline_agent.qnetwork_local = load_model(baseline_agent.qnetwork_local, baseline_model_path)

            ndlinear_agent = DQNAgent(state_size, action_size, use_ndlinear=True, seed=args.seed)
            ndlinear_model_path = f"models/ndlinear/{env_name_short}_dqn_ndlinear.pth"
            ndlinear_agent.qnetwork_local = load_model(ndlinear_agent.qnetwork_local, ndlinear_model_path)

            baseline_results, ndlinear_results = run_all_benchmarks(baseline_agent, ndlinear_agent, env)
            # Plot benchmark results
            plot_inference_time(baseline_results['inference_time'], ndlinear_results['inference_time'])
            plot_memory_usage(baseline_results['memory_usage'], ndlinear_results['memory_usage'])
            plot_model_size(baseline_results['model_size'], ndlinear_results['model_size'])

        else:
            print("Benchmarking requires both standard and NdLinear models.")

    elif args.mode == 'record':
        if args.env == "Pendulum-v1":
            print(f"Recording a PPO rollout from {PENDULUM_CONFIG['model_path']} …")
            model = PPO.load(PENDULUM_CONFIG['model_path'])
            class PPOAgentWrapper:
                def __init__(self, model):
                    self.model = model
                def select_action(self, obs):
                    action, _ = self.model.predict(obs, deterministic=True)
                    return action
            ppo_agent = PPOAgentWrapper(model)
            record_episode(ppo_agent, config['env_name'], num_episodes=args.num_episodes)
            # record_env = gym.make(PENDULUM_CONFIG['env_name'], render_mode="human")
            # model = PPO.load(PENDULUM_CONFIG['model_path'], env=record_env)
            # obs, _ = gym.make("Pendulum-v1").reset()
            # obs, _ = record_env.reset()
            # for _ in range(5):
            #     action, _ = model.predict(obs, deterministic=True)
            #     obs, _, terminated, truncated, _ = record_env.step(action)
            #     # record_env.render()
            #     if terminated or truncated:
            #         obs, _ = record_env.reset()
            # record_env.close()
            return

        agent_to_record = None
        if args.model == 'standard':
            agent_to_record = DQNAgent(state_size, action_size, use_ndlinear=False, seed=args.seed)
            model_path = f"models/baseline/{env_name_short}_dqn_baseline.pth"
            agent_to_record.qnetwork_local = load_model(agent_to_record.qnetwork_local, model_path)
        elif args.model == 'ndlinear':
            agent_to_record = DQNAgent(state_size, action_size, use_ndlinear=True, seed=args.seed)
            model_path = f"models/ndlinear/{env_name_short}_dqn_ndlinear.pth"
            agent_to_record.qnetwork_local = load_model(agent_to_record.qnetwork_local, model_path)
        else:
            print("Recording requires specifying either 'standard' or 'ndlinear' model.")
            return

        if agent_to_record:
            record_episode(agent_to_record, config['env_name'], num_episodes=args.num_episodes)


    if args.mode != 'train':
        env.close()


if __name__ == "__main__":
    main()