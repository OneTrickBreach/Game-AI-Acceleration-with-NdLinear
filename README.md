# Game AI Acceleration with NdLinear

## Project Overview

This project aims to benchmark the performance of standard Deep Q-Networks (DQN) against NdLinear-optimized DQNs in classic control game environments. It provides a framework for training, evaluating, benchmarking, and recording the gameplay of these agents.

## Objectives

*   Create a comprehensive benchmarking framework for comparing standard neural networks with NdLinear-optimized equivalents.
*   Measure and document performance improvements across multiple metrics and game environments.
*   Analyze the trade-offs between model size, inference speed, and game performance.
*   Produce visualizations that clearly demonstrate the advantages of NdLinear optimization.
*   Document the implementation process and results for reproducibility.

## Gymnasium Environments

This project utilizes the following classic control environments from Gymnasium:

*   **CartPole-v1:** The goal is to balance a pole on a moving cart. It has a discrete action space (move left or right) and a 4-dimensional observation space.
*   **MountainCar-v0:** The goal is to drive an underpowered car up a steep hill to reach a flag. It has a discrete action space (accelerate left, accelerate right, or do nothing) and a 2-dimensional observation space.
*   **Pendulum-v1:** The goal is to swing a pendulum up and keep it upright. It has a continuous action space (torque) and a 3-dimensional observation space. Note that DQN is not ideally suited for continuous action spaces, and performance may be limited.
*   **Acrobot-v1:** The goal is to swing a two-link robot arm to reach a target height. It has a discrete action space (apply torque to the joints) and a 6-dimensional observation space.

## Methodology

### Environment Setup

*   Use Gymnasium's classic control environments (CartPole-v1, MountainCar-v0, Pendulum-v1, and Acrobot-v1).
*   Implement in Python with PyTorch as the primary deep learning framework.
*   Set up consistent evaluation metrics and benchmark procedures.

### Model Development

1.  **Baseline Models:**
    *   Implement standard reinforcement learning agents (DQN) with traditional `nn.Linear` layers.
    *   Train these models to achieve good performance on each environment.

2.  **NdLinear Models:**
    *   Create identical network architectures but replace `nn.Linear` with NdLinear implementations.
    *   Train these models with the same hyperparameters and random seeds.

### Benchmarking Framework

Develop a robust benchmarking system that measures:

*   Model size (parameters and storage requirements).
*   Inference time (average decision time per step).
*   Memory usage during training and inference.
*   Training efficiency (episodes to convergence).
*   Game performance (average rewards/scores).
*   Maximum achievable frame rate.

### Visualization and Analysis

*   Generate performance comparison graphs across all metrics.
*   Create visual demonstrations of both models playing the games.
*   Analyze potential use cases where NdLinear provides the most significant advantages.

## Installation Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ensemble-core/Game-AI-Acceleration.git
    cd Game-AI-Acceleration
    ```

2.  **Create a Python virtual environment with Python 3.12:**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    ```bash
    .venv\Scripts\activate
    ```

4.  **Install the required packages:**

    ```bash
    pip install gymnasium matplotlib numpy pandas tqdm
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

5.  **Clone and install the NdLinear package:**

    ```bash
    git clone https://github.com/ensemble-core/NdLinear.git
    cd NdLinear
    pip install .
    cd ..
    ```

## Project Structure

The project has the following directory structure:

```
Game AI Acceleration/
├── benchmarks/              # Benchmarking scripts
├── data/                    # For storing training results (not currently used)
├── models/                  # For saving trained models
│   ├── baseline/            # Trained baseline DQN models
│   └── ndlinear/            # Trained NdLinear DQN models
├── plots/                   # Generated performance plots
│   ├── inference_time.png   # Inference time comparison plot
│   ├── memory_usage.png     # Memory usage comparison plot
│   └── model_size.png       # Model size comparison plot
├── recordings/              # Recorded agent gameplay videos
│   └── <environment>/       # Recordings for each environment
│       └── <model>_episode_*.mp4 # Individual episode recordings
├── src/                     # Source code
│   ├── agents/              # Agent implementations (DQN)
│   ├── environments/        # Environment wrappers (GymEnvWrapper)
│   ├── models/              # Neural network architectures (DQN, NdDQN)
│   ├── training/            # Training scripts (cartpole, mountaincar, pendulum, acrobot)
│   └── utils/               # Utility functions (experiment config, model utils, replay buffer)
├── main.py                  # Main execution script
├── demo.py                  # Script for demonstrating trained agents
├── plan.md                  # Project roadmap
├── proposal.md              # Project proposal
├── README.md                # This file
└── requirements.txt         # List of Python dependencies
```

## Usage Examples

To train a DQN agent on the CartPole-v1 environment, run the following command:

```bash
python main.py --mode train --env CartPole-v1
```

To evaluate a trained agent, use the 'evaluate' mode:

```bash
python main.py --mode evaluate --env CartPole-v1 --model both
```

To benchmark the performance of the trained agents, use the 'benchmark' mode:

```bash
python main.py --mode benchmark --env CartPole-v1 --model both
```

To record the gameplay of a trained agent, use the 'record' mode:

```bash
python main.py --mode record --env CartPole-v1 --model standard
```

To demonstrate a trained agent in a game environment, run the following command:

```bash
python demo.py --env <environment_name> --model_path <path_to_model> --render
```

*   `<environment_name>`: The name of the environment (e.g., CartPole-v1, MountainCar-v0, Pendulum-v1, Acrobot-v1).
*   `<path_to_model>`: The path to the trained model (.pth file) you want to demonstrate.
*   `--render`: An optional flag to enable rendering of the environment during gameplay.

For example, to demonstrate the trained baseline DQN agent on the CartPole-v1 environment with rendering enabled, you would run:

```bash
python demo.py --env CartPole-v1 --model_path models/baseline/cartpole_dqn_baseline.pth --render
```

## Results and Findings

Based on the benchmarking results, the NdLinear-optimized DQN agents generally exhibit the following characteristics compared to the baseline DQN agents:

*   **Lower Inference Time:** The NdLinear agents typically have a faster inference time due to the compressed model structure.
*   **Lower Memory Usage:** The NdLinear agents consume less memory due to the reduced number of parameters and smaller model size.
*   **Smaller Model Size:** The NdLinear models have a significantly smaller file size compared to the baseline models, making them more efficient for storage and deployment.

(Note: Specific numerical results and performance comparisons will be added here after a more thorough analysis of the generated plots and data.)

## Benchmarking Methodology

The benchmarking process involves the following steps:

1.  **Training:** Train both the baseline DQN agent and the NdLinear-optimized DQN agent on the same environment using the same training parameters.
2.  **Inference Time Measurement:** Measure the average time it takes for each agent to select an action in the environment over a fixed number of steps. This is done using the `measure_inference_time` function in `benchmarks/benchmark_utils.py`.
3.  **Memory Usage Measurement:** Measure the memory usage of each agent's model using the `measure_memory_usage` function in `benchmarks/benchmark_utils.py`. This measures the resident set size (RSS) of the process, which represents the amount of memory allocated to the model.
4.  **Model Size Measurement:** Measure the size of each agent's saved model file using the `measure_model_size` function in `benchmarks/benchmark_utils.py`. This involves saving the model's state dictionary to a temporary file and then measuring the file size.
5.  **Performance Comparison:** Compare the results for each metric (inference time, memory usage, and model size) between the baseline and NdLinear agents.

## Next Steps

*   Implement a more suitable agent for continuous control environments like Pendulum (e.g., PPO or DDPG).
*   Add more detailed experiment configurations and hyperparameters to `src/utils/experiment_config.py`.
*   Refine the benchmarking and visualization implementations as needed.
