CARTPOLE_CONFIG = {
    'env_name': 'CartPole-v1',
    'max_episodes': 1000,
    'target_reward': 195.0,
    'hyperparameters': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 10000,
        'update_every': 4
    }
}

MOUNTAINCAR_CONFIG = {
    'env_name': 'MountainCar-v0',
    'max_episodes': 1000,
    'target_reward': -110.0, # Example target reward for MountainCar
    'hyperparameters': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 10000,
        'update_every': 4
    }
}

PENDULUM_CONFIG = {
    'env_name': 'Pendulum-v1',
    'total_timesteps': 200_000,
    'model_path': 'models/ppo/pendulum_ppo.zip',
    'target_reward': -200.0,
    'hyperparameters': {
        'learning_rate': 5e-4,    # your original 0.0005
        'n_steps':       2048,    # rollout length per update
        'batch_size':    128,     # minibatch size for each epoch
        'gamma':         0.99,    # discount factor
        'ent_coef':      0.0,     # entropy bonus coefficient
    }
}


ACROBOT_CONFIG = {
    'env_name': 'Acrobot-v1',
    'max_episodes': 1000,
    'target_reward': -100.0, # Example target reward for Acrobot (solving in fewer than 100 steps, reward is -1 per step)
    'hyperparameters': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 10000,
        'update_every': 4
    }
}