import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.experiment_config import PENDULUM_CONFIG

def train_pendulum():
    """Trains a PPO agent on Pendulum-v1 using Stable-Baselines3."""
    print("Training agents on Pendulum-v1 environment with PPO...")

    # Load config
    config = PENDULUM_CONFIG
    env_id = config.get('env_name', 'Pendulum-v1')

    # Create a vectorized env for SB3
    env = DummyVecEnv([lambda: gym.make(env_id)])

    # Pull PPO hyperparameters (with defaults)
    hyper = config.get('hyperparameters', {})
    ppo_kwargs = {
        'learning_rate': hyper.get('learning_rate', 3e-4),
        'n_steps':        hyper.get('n_steps', 2048),
        'batch_size':     hyper.get('batch_size', 64),
        'gamma':          hyper.get('gamma', 0.99),
        'ent_coef':       hyper.get('ent_coef', 0.0),
        'verbose':        1,
    }

    # Initialize and train
    model = PPO('MlpPolicy', env, **ppo_kwargs)
    total_timesteps = config.get('total_timesteps', 100_000)
    print(f"→ Learning for {total_timesteps} timesteps with parameters: {ppo_kwargs}")
    model.learn(total_timesteps=total_timesteps)

    # Save the trained policy
    save_dir = config.get('model_path', f"models/ppo/{env_id}.zip")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model.save(save_dir)
    print(f"PPO model saved to {save_dir}")

    # (Optional) quick eval demo
    print("\nRunning a few evaluation episodes:")
    eval_env = gym.make(env_id)
    obs, _ = eval_env.reset()
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        eval_env.render()
        if terminated or truncated:
            obs, _ = eval_env.reset()
    eval_env.close()

    env.close()
    print("\nPendulum-v1 PPO training complete.")
