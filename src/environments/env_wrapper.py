import gymnasium as gym
import numpy as np

class GymEnvWrapper:
    def __init__(self, env_name, max_steps=500, render_mode=None):
        if render_mode:
            self.env = gym.make(env_name, render_mode=render_mode)
        else:
            self.env = gym.make(env_name)
        self.max_steps = max_steps

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info

    def close(self):
        self.env.close()