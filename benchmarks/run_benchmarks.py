from benchmarks.benchmark_utils import measure_inference_time, measure_memory_usage, count_parameters, measure_model_size
from visualization.performance_plots import (plot_inference_time, plot_memory_usage, plot_model_size)
import os

def run_all_benchmarks(baseline_agent, ndlinear_agent, env, num_steps=1000):
    """Runs all benchmarks for baseline and NdLinear agents."""
    print("\nRunning benchmarks...")

    baseline_results = {}
    ndlinear_results = {}

    # Measure inference time
    baseline_results['inference_time'] = measure_inference_time(baseline_agent, env, num_steps)
    ndlinear_results['inference_time'] = measure_inference_time(ndlinear_agent, env, num_steps)

    # Measure memory usage
    baseline_results['memory_usage'] = measure_memory_usage(baseline_agent)
    ndlinear_results['memory_usage'] = measure_memory_usage(ndlinear_agent)

    # Count parameters
    baseline_results['parameter_count'] = count_parameters(baseline_agent.qnetwork_local)
    ndlinear_results['parameter_count'] = count_parameters(ndlinear_agent.qnetwork_local)

    # Measure model size
    baseline_results['model_size'] = measure_model_size(baseline_agent.qnetwork_local, filename="baseline_model_size.pth")
    ndlinear_results['model_size'] = measure_model_size(ndlinear_agent.qnetwork_local, filename="ndlinear_model_size.pth")

    os.makedirs("plots", exist_ok=True)
    plot_inference_time(
        baseline_results['inference_time'],
        ndlinear_results['inference_time'],
        save_path="plots/inference_time.png",
    )
    print("Saved inference_time.png in plots/")
    plot_memory_usage(
        baseline_results['memory_usage'],
        ndlinear_results['memory_usage'],
        save_path="plots/memory_usage.png",
    )
    print("Saved memory_usage.png in plots/")
    plot_model_size(
        baseline_results['model_size'],
        ndlinear_results['model_size'],
        save_path="plots/model_size.png",
    )
    print("Saved model_size.png in plots/")
    print("\nBenchmarks complete.")
    return baseline_results, ndlinear_results