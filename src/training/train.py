import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from src.environments.env_wrapper import GymEnvWrapper
from src.agents.dqn_agent import DQNAgent
from src.utils.model_utils import save_model # Import save_model

def train_agent(agent, env, num_episodes=1000, max_t=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, save_filename=None, save_every=100):
    """Deep Q-Learning.

    Params
    ======
        agent (DQNAgent): the agent to train
        env (GymEnvWrapper): the environment to train in
        num_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        epsilon_start (float): starting value of epsilon, for epsilon-greedy action selection
        epsilon_end (float): minimum value of epsilon
        epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
        save_filename (str): filename to save the model (optional)
        save_every (int): save the model every 'save_every' episodes
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    epsilon = epsilon_start              # initialize epsilon

    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        epsilon = max(epsilon_end, epsilon_decay*epsilon) # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        # Save model periodically
        if save_filename and i_episode % save_every == 0:
            save_model(agent.qnetwork_local, save_filename)

        if np.mean(scores_window) >= 200.0: # Example success criterion, should be adjusted per environment
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # Save model when environment is solved
            if save_filename:
                save_model(agent.qnetwork_local, save_filename)
            break


    return scores

def evaluate_agent(agent, env, num_episodes=100, max_t=1000, render=False):
    """Evaluates the performance of a trained agent.

    Params
    ======
        agent (DQNAgent): the trained agent to evaluate
        env (GymEnvWrapper): the environment to evaluate in
        num_episodes (int): number of evaluation episodes
        max_t (int): maximum number of timesteps per episode
        render (bool): whether to render the environment
    """
    scores = []
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.select_action(state, epsilon=0.0) # Use greedy policy for evaluation
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if render:
                env.render()
            if done:
                break
        scores.append(score)
        print('\rEvaluation Episode {}\tScore: {:.2f}'.format(i_episode, score), end="")

    avg_score = np.mean(scores)
    print('\nAverage evaluation score over {} episodes: {:.2f}'.format(num_episodes, avg_score))
    return scores