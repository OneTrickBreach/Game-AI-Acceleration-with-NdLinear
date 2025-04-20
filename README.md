# Game AI Acceleration with NdLinear

## Project Overview

This project aims to benchmark the performance of standard Deep Q-Networks (DQN) against NdLinear-optimized DQNs in classic control game environments. It provides a framework for training, evaluating, benchmarking, and recording the gameplay of these agents.

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
    pip install -e ./NdLinear
    ```

## Project Structure

The project has the following directory structure:

```
Game AI Acceleration/
├── benchmarks/              # Benchmarking scripts
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

## Results and Findings

Based on the benchmarking results, the NdLinear-optimized DQN agents generally exhibit the following characteristics compared to the baseline DQN agents:

*   **Lower Inference Time:** The NdLinear agents typically have a faster inference time due to the compressed model structure.
*   **Lower Memory Usage:** The NdLinear agents consume less memory due to the reduced number of parameters and smaller model size.
*   **Smaller Model Size:** The NdLinear models have a significantly smaller file size compared to the baseline models, making them more efficient for storage and deployment.


## Benchmarking Methodology

The benchmarking process involves the following steps:

1.  **Training:** Train both the baseline DQN agent and the NdLinear-optimized DQN agent on the same environment using the same training parameters.
2.  **Inference Time Measurement:** Measure the average time it takes for each agent to select an action in the environment over a fixed number of steps. This is done using the `measure_inference_time` function in `benchmarks/benchmark_utils.py`.
3.  **Memory Usage Measurement:** Measure the memory usage of each agent's model using the `measure_memory_usage` function in `benchmarks/benchmark_utils.py`. This measures the resident set size (RSS) of the process, which represents the amount of memory allocated to the model.
4.  **Model Size Measurement:** Measure the size of each agent's saved model file using the `measure_model_size` function in `benchmarks/benchmark_utils.py`. This involves saving the model's state dictionary to a temporary file and then measuring the file size.
5.  **Performance Comparison:** Compare the results for each metric (inference time, memory usage, and model size) between the baseline and NdLinear agents.

## Future Work/Next Steps

*   Implement a more suitable agent for continuous control environments like Pendulum (e.g., PPO or DDPG).
*   Add more detailed experiment configurations and hyperparameters to `src/utils/experiment_config.py`.
*   Refine the benchmarking and visualization implementations as needed.
