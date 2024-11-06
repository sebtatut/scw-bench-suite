import os
import subprocess
import re
import statistics
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

def parse_output(output: str) -> Dict[str, float]:
    """Extracts metrics from benchmark output, handling missing data gracefully."""
    def extract_metric(pattern: str) -> float:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Expected pattern '{pattern}' not found in output.")

    metrics = {
        'Output token throughput': extract_metric(r"Output token throughput \(tok/s\):\s+([\d.]+)"),
        'Median TTFT': extract_metric(r"Median TTFT \(ms\):\s+([\d.]+)"),
        'Median TPOT': extract_metric(r"Median TPOT \(ms\):\s+([\d.]+)"),
        'Median ITL': extract_metric(r"Median ITL \(ms\):\s+([\d.]+)")
    }
    return metrics


def collect_metrics(args: argparse.Namespace) -> Dict[int, Dict[str, float]]:
    """Runs the benchmark for each concurrency level and collects median metrics."""
    all_results: Dict[int, Dict[str, float]] = {}

    for concurrency_level in range(args.concurrency_level_start, args.concurrency_level_end + 1):
        metrics_per_run: Dict[str, List[float]] = {
            'Output token throughput': [],
            'Median TTFT': [],
            'Median TPOT': [],
            'Median ITL': []
        }

        for _ in range(args.runs):
            # run bench
            cmd: List[str] = [
                "python3", "/repos/vllm/benchmarks/benchmark_serving.py",
                "--backend", args.backend,
                "--dataset-name", args.dataset_name,
                "--model", args.model,
                "--num-prompts", str(concurrency_level),
                "--endpoint", args.endpoint,
                "--tokenizer", args.tokenizer,
                "--random-input-len", str(args.random_input_len),
                "--random-output-len", str(args.random_output_len),
                "--disable-tqdm"
            ]

            # print cmd for debugging
            print("Executing command:", " ".join(cmd))

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error: Command failed with exit code {result.returncode}")
                print(result.stderr)
                raise RuntimeError("Benchmarking script failed. Stopping further execution.")

            metrics = parse_output(result.stdout)

            # store each run's metrics
            for key, value in metrics.items():
                metrics_per_run[key].append(value)

        # calculate median for each metric -> store
        all_results[concurrency_level] = {
            metric: statistics.median(values)
            for metric, values in metrics_per_run.items()
        }

    return all_results

def plot_metrics(results: Dict[int, Dict[str, float]], save_plot: Optional[str] = None, plot_dim: Optional[Tuple[int, int]] = None) -> None:
    """Plots the metrics using the memorized plotting procedure."""
    
    # extract concurrency levels and metrics for plotting
    concurrency_levels = list(results.keys())
    output_token_throughput = [results[level]['Output token throughput'] for level in concurrency_levels]
    median_ttft = [results[level]['Median TTFT'] for level in concurrency_levels]
    median_tpot = [results[level]['Median TPOT'] for level in concurrency_levels]
    median_itl = [results[level]['Median ITL'] for level in concurrency_levels]

    # normalize each metric list to improve visibility on a common scale
    metrics = [output_token_throughput, median_ttft, median_tpot, median_itl]
    normalized_metrics: List[List[float]] = []
    for metric in metrics:
        min_val, max_val = min(metric), max(metric)
        # normalize and handle cases where max_val == min_val to avoid division by zero
        normalized_metrics.append([(val - min_val) / (max_val - min_val) if max_val != min_val else 1 for val in metric])

    # configure the plot dimensions
    plt.figure(figsize=(plot_dim[0] / 100, plot_dim[1] / 100) if plot_dim else (10, 6)) # type: ignore
    
    # colors and markers for each metric
    plt.plot(concurrency_levels, normalized_metrics[0], marker='o', label='Output Token Throughput (tok/s)', color='blue') # type: ignore
    plt.plot(concurrency_levels, normalized_metrics[1], marker='s', label='Median TTFT (ms)', color='red') # type: ignore
    plt.plot(concurrency_levels, normalized_metrics[2], marker='^', label='Median TPOT (ms)', color='green') # type: ignore
    plt.plot(concurrency_levels, normalized_metrics[3], marker='D', label='Median ITL (ms)', color='purple') # type: ignore
    
    # axes and title labels
    plt.xlabel('Concurrency Level') # type: ignore
    plt.ylabel('Normalized Metrics') # type: ignore
    plt.title('Normalized Benchmark Metrics by Concurrency Level') # type: ignore
    
    # legend and grid
    plt.legend() # type: ignore
    plt.grid(True) # type: ignore

    # save | display the plot
    if save_plot:
        current_dir = os.getcwd()
        save_path = f"{current_dir}/benchmark_metrics.{save_plot.lower()}"
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, format=save_plot.lower(), dpi=100) # type: ignore
    else:
        plt.show() # type: ignore


def plot_metrics_trellis(results: Dict[int, Dict[str, float]], save_plot: Optional[str] = None, plot_dim: Optional[Tuple[int, int]] = None) -> None:
    """Plots the metrics using a trellis of bar charts with trend lines."""
    
    # extract concurrency levels and metrics for plotting
    concurrency_levels = list(results.keys())
    output_token_throughput = [results[level]['Output token throughput'] for level in concurrency_levels]
    median_ttft = [results[level]['Median TTFT'] for level in concurrency_levels]
    median_tpot = [results[level]['Median TPOT'] for level in concurrency_levels]
    median_itl = [results[level]['Median ITL'] for level in concurrency_levels]
    
    # setup figure dimensions
    fig_size = (plot_dim[0] / 100, plot_dim[1] / 100) if plot_dim else (14, 10)
    _, axes = plt.subplots(2, 2, figsize=fig_size) # type: ignore
    axes = axes.flatten() # type: ignore

    # map metric to its respective position in the grid
    metrics_data = { # type: ignore
        'Output Token Throughput': {'values': output_token_throughput, 'color': 'red', 'axis': axes[0]},
        'Median TTFT': {'values': median_ttft, 'color': 'blue', 'axis': axes[1]},
        'Median TPOT': {'values': median_tpot, 'color': 'green', 'axis': axes[2]},
        'Median ITL': {'values': median_itl, 'color': 'purple', 'axis': axes[3]}
    }

    # create bar charts with trend lines for each metric
    for metric_name, properties in metrics_data.items(): # type: ignore
        values = properties['values'] # type: ignore
        ax = properties['axis'] # type: ignore
        color = properties['color'] # type: ignore
        
        # bar chart
        ax.bar(concurrency_levels, values, color=color, alpha=0.6, label=metric_name) # type: ignore
        
        # add trend line
        ax.plot(concurrency_levels, values, marker='o', color=color, linestyle='-', linewidth=1.5) # type: ignore
        
        # setup labels and title
        ax.set_title(metric_name) # type: ignore
        ax.set_xlabel("Concurrency Level") # type: ignore
        ax.set_ylabel(metric_name) # type: ignore
        ax.grid(True, which='both', linestyle='--', linewidth=0.5) # type: ignore

    # setup layout
    plt.suptitle("Benchmark Metrics by Concurrency Level", fontsize=16) # type: ignore
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
    
    if save_plot:
        current_dir = os.getcwd()
        save_path = f"{current_dir}/benchmark_metrics_trellis.{save_plot.lower()}"
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, format=save_plot.lower(), dpi=100) # type: ignore
    else:
        plt.show() # type: ignore


def plot_metrics_trellis_line_only(results: Dict[int, Dict[str, float]], save_plot: Optional[str] = None, plot_dim: Optional[Tuple[int, int]] = None) -> None:
    """Plots the metrics using a trellis of line charts only."""
    
    # extract concurrency levels and metrics for plotting
    concurrency_levels = list(results.keys())
    output_token_throughput = [results[level]['Output token throughput'] for level in concurrency_levels]
    median_ttft = [results[level]['Median TTFT'] for level in concurrency_levels]
    median_tpot = [results[level]['Median TPOT'] for level in concurrency_levels]
    median_itl = [results[level]['Median ITL'] for level in concurrency_levels]
    
    # setup figure dimensions
    fig_size = (plot_dim[0] / 100, plot_dim[1] / 100) if plot_dim else (14, 10)
    _, axes = plt.subplots(2, 2, figsize=fig_size) # type: ignore
    axes = axes.flatten() # type: ignore

    # map each metric to its respective position in the grid
    metrics_data = { # type: ignore
        'Output Token Throughput': {'values': output_token_throughput, 'color': 'red', 'axis': axes[0]},
        'Median TTFT': {'values': median_ttft, 'color': 'blue', 'axis': axes[1]},
        'Median TPOT': {'values': median_tpot, 'color': 'green', 'axis': axes[2]},
        'Median ITL': {'values': median_itl, 'color': 'purple', 'axis': axes[3]}
    }

    # create line charts for each metric
    for metric_name, properties in metrics_data.items(): # type: ignore
        values = properties['values'] # type: ignore
        ax = properties['axis'] # type: ignore
        color = properties['color'] # type: ignore
        
        # line chart
        ax.plot(concurrency_levels, values, marker='o', color=color, linestyle='-', linewidth=1.5, markersize=4) # type: ignore
        
        # setup labels + title
        ax.set_title(metric_name) # type: ignore
        ax.set_xlabel("Concurrency Level") # type: ignore
        ax.set_ylabel(metric_name) # type: ignore
        ax.grid(True, which='both', linestyle='--', linewidth=0.5) # type: ignore

    # setup layout
    plt.suptitle("Benchmark Metrics by Concurrency Level", fontsize=16) # type: ignore
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
    
    if save_plot:
        current_dir = os.getcwd()
        save_path = f"{current_dir}/benchmark_metrics_trellis_line.{save_plot.lower()}"
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, format=save_plot.lower(), dpi=100) # type: ignore
    else:
        plt.show() # type: ignore



def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark tests at various concurrency levels and collect metrics.")
    parser.add_argument("--backend", required=True, help="Backend name (e.g., vllm)")
    parser.add_argument("--dataset-name", required=True, help="Dataset name (e.g., random)")
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--endpoint", required=True, help="Endpoint URL")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name")
    parser.add_argument("--random-input-len", type=int, default=4096, help="Random input length")
    parser.add_argument("--random-output-len", type=int, default=1024, help="Random output length")
    parser.add_argument("--disable-tqdm", action='store_true', help="Disable tqdm")
    parser.add_argument("--concurrency-level-start", type=int, required=True, help="Starting concurrency level")
    parser.add_argument("--concurrency-level-end", type=int, required=True, help="Ending concurrency level")
    parser.add_argument("--runs", type=int, required=True, help="Number of runs per concurrency level")
    parser.add_argument("--save-plot", choices=['PNG', 'SVG'], help="Save plot as PNG or SVG")
    parser.add_argument("--plot-dim", type=str, help="Plot dimensions in the form 'width x height' (e.g., 800x600)")

    args = parser.parse_args()

    # parse plot dimensions if provided
    plot_dim: Optional[Tuple[int, int]] = None
    if args.plot_dim:
        plot_dim = tuple(map(int, args.plot_dim.split('x'))) # type: ignore

    # collect metrics
    results = collect_metrics(args)

    # print medians by concurrency level
    for level, metrics in results.items():
        print(f"Concurrency Level {level}:")
        for metric, median in metrics.items():
            print(f"  {metric}: {median}")

    # plot if required
    if args.save_plot:
        plot_metrics(results, save_plot=args.save_plot, plot_dim=plot_dim)
        plot_metrics_trellis(results, save_plot=args.save_plot, plot_dim=plot_dim)
        plot_metrics_trellis_line_only(results, save_plot=args.save_plot, plot_dim=plot_dim)

if __name__ == "__main__":
    main()
