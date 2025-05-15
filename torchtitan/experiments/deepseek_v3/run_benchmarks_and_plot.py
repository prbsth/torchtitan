import os
import json
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_single_benchmark(num_gpus, experts_per_gpu, num_layers, batch_size, seq_len, iterations, use_overlap=False):
    """Run a single benchmark with specified parameters"""
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={num_gpus}",
        "benchmark_moe_overlap.py",
        f"--num_gpus={num_gpus}",
        f"--experts_per_gpu={experts_per_gpu}",
        f"--num_layers={num_layers}",
        f"--batch_size={batch_size}",
        f"--seq_len={seq_len}",
        f"--iterations={iterations}"
    ]
    
    if use_overlap:
        cmd.append("--use_overlap")
    
    mode = "With Overlap" if use_overlap else "Without Overlap"
    print(f"\n{'='*80}")
    print(f"Running benchmark {mode} with sequence length: {seq_len}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run the command and capture all output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    # Write raw output to a file for debugging
    output_filename = f"benchmark_output_seq_{seq_len}_{mode.replace(' ', '_')}.txt"
    with open(output_filename, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
    
    # Parse results
    perf_time = None
    throughput = None
    output = result.stdout
    
    if use_overlap:
        time_line = [line for line in output.split('\n') if "Comet (with overlap):" in line]
        throughput_line = [line for line in output.split('\n') if "Tokens/sec Comet:" in line]
    else:
        time_line = [line for line in output.split('\n') if "Vanilla (no overlap):" in line]
        throughput_line = [line for line in output.split('\n') if "Tokens/sec vanilla:" in line]
    
    if time_line:
        try:
            perf_time = float(time_line[0].split(":")[1].strip().split()[0])
        except (IndexError, ValueError) as e:
            print(f"Error parsing time: {e}")
    
    if throughput_line:
        try:
            throughput = float(throughput_line[0].split(":")[1].strip())
        except (IndexError, ValueError) as e:
            print(f"Error parsing throughput: {e}")
    
    # If we couldn't find the results, provide default values
    if perf_time is None:
        print(f"WARNING: Could not parse execution time for seq_len={seq_len}, {mode}")
        perf_time = 0.0
    
    if throughput is None:
        print(f"WARNING: Could not parse throughput for seq_len={seq_len}, {mode}")
        throughput = 0.0
    
    return {"time": perf_time, "throughput": throughput}

def run_benchmark_pair(num_gpus, experts_per_gpu, num_layers, batch_size, seq_len, iterations):
    """Run both vanilla and overlap benchmarks for a given sequence length"""
    # First run without overlap (vanilla)
    vanilla_results = run_single_benchmark(
        num_gpus, experts_per_gpu, num_layers, batch_size, seq_len, iterations, use_overlap=False
    )
    
    # Then run with overlap
    overlap_results = run_single_benchmark(
        num_gpus, experts_per_gpu, num_layers, batch_size, seq_len, iterations, use_overlap=True
    )
    
    # Calculate speedup and improvement
    if vanilla_results["time"] > 0 and overlap_results["time"] > 0:
        speedup = vanilla_results["time"] / overlap_results["time"]
        improvement = (vanilla_results["time"] - overlap_results["time"]) / vanilla_results["time"] * 100
    else:
        speedup = 1.0
        improvement = 0.0
    
    return {
        "seq_len": seq_len,
        "vanilla_time": vanilla_results["time"],
        "overlap_time": overlap_results["time"],
        "speedup": speedup,
        "improvement": improvement,
        "vanilla_tokens_per_sec": vanilla_results["throughput"],
        "overlap_tokens_per_sec": overlap_results["throughput"]
    }

def create_plots(results, output_dir):
    """Create and save plots from the benchmark results"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    seq_lengths = [result["seq_len"] for result in results]
    
    # For consistency in plots
    seq_lengths_labels = [str(seq_len) for seq_len in seq_lengths]
    
    # Times plot (ms/iter)
    vanilla_times = [result.get("vanilla_time", 0) for result in results]
    overlap_times = [result.get("overlap_time", 0) for result in results]
    
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    x = np.arange(len(seq_lengths))
    
    plt.bar(x - bar_width/2, vanilla_times, bar_width, label='Traditional (no overlap)')
    plt.bar(x + bar_width/2, overlap_times, bar_width, label='Comet (with overlap)')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms/iter)')
    plt.title('Benchmark Times Comparison')
    plt.xticks(x, seq_lengths_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add values on top of the bars
    for i, v in enumerate(vanilla_times):
        if v > 0:
            plt.text(i - bar_width/2, v + 1, f"{v:.1f}", ha='center')
    for i, v in enumerate(overlap_times):
        if v > 0:
            plt.text(i + bar_width/2, v + 1, f"{v:.1f}", ha='center')
    
    plt.savefig(f"{output_dir}/time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved time comparison plot to {output_dir}/time_comparison.png")
    
    # Speedup plot
    plt.figure(figsize=(10, 6))
    speedups = [result.get("speedup", 1.0) for result in results]
    improvements = [result.get("improvement", 0.0) for result in results]
    
    plt.bar(x, speedups, width=0.6, color='green')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup (×)')
    plt.title('Speedup Factor (Traditional vs. Comet)')
    plt.xticks(x, seq_lengths_labels)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    # Add values on top of the bars
    for i, v in enumerate(speedups):
        if v > 0:
            plt.text(i, v + 0.05, f"{v:.2f}×", ha='center')
    
    plt.savefig(f"{output_dir}/speedup_factor.png", dpi=300, bbox_inches='tight')
    print(f"Saved speedup plot to {output_dir}/speedup_factor.png")
    
    # Throughput plot (tokens/sec)
    plt.figure(figsize=(12, 8))
    vanilla_throughput = [result.get("vanilla_tokens_per_sec", 0) for result in results]
    overlap_throughput = [result.get("overlap_tokens_per_sec", 0) for result in results]
    
    plt.bar(x - bar_width/2, vanilla_throughput, bar_width, label='Traditional (no overlap)')
    plt.bar(x + bar_width/2, overlap_throughput, bar_width, label='Comet (with overlap)')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Throughput Comparison')
    plt.xticks(x, seq_lengths_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add values on top of the bars
    max_throughput = max(max(vanilla_throughput) if vanilla_throughput else 0, 
                         max(overlap_throughput) if overlap_throughput else 0)
    offset = max_throughput * 0.02 if max_throughput > 0 else 1
    
    for i, v in enumerate(vanilla_throughput):
        if v > 0:
            plt.text(i - bar_width/2, v + offset, f"{int(v)}", ha='center')
    for i, v in enumerate(overlap_throughput):
        if v > 0:
            plt.text(i + bar_width/2, v + offset, f"{int(v)}", ha='center')
    
    plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved throughput plot to {output_dir}/throughput_comparison.png")
    
    # Create a summary plot combining key metrics
    plt.figure(figsize=(14, 10))
    
    # Add improvement percentage labels
    improvement_labels = [f"{imp:.1f}%" for imp in improvements]
    
    # Create 4 subplots
    plt.subplot(2, 2, 1)
    plt.bar(x - bar_width/2, vanilla_times, bar_width, label='Traditional')
    plt.bar(x + bar_width/2, overlap_times, bar_width, label='Comet')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms/iter)')
    plt.title('Execution Time')
    plt.xticks(x, seq_lengths_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.bar(x, speedups, width=0.6, color='green')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup (Traditional vs. Comet)')
    plt.xticks(x, seq_lengths_labels)
    for i, (v, imp) in enumerate(zip(speedups, improvement_labels)):
        if v > 0:
            plt.text(i, v + 0.05, f"{v:.2f}×\n{imp}", ha='center')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    plt.subplot(2, 2, 3)
    plt.bar(x - bar_width/2, vanilla_throughput, bar_width, label='Traditional')
    plt.bar(x + bar_width/2, overlap_throughput, bar_width, label='Comet')
    plt.xlabel('Sequence Length')
    plt.ylabel('Tokens/sec')
    plt.title('Throughput')
    plt.xticks(x, seq_lengths_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Calculate relative throughput (handle division by zero)
    relative_throughput = []
    for o, v in zip(overlap_throughput, vanilla_throughput):
        if v > 0:
            relative_throughput.append(o / v)
        else:
            relative_throughput.append(1.0)
    
    plt.bar(x, relative_throughput, width=0.6, color='purple')
    plt.xlabel('Sequence Length')
    plt.ylabel('Relative Throughput')
    plt.title('Throughput Improvement')
    plt.xticks(x, seq_lengths_labels)
    for i, v in enumerate(relative_throughput):
        if v > 0:
            plt.text(i, v + 0.05, f"{v:.2f}×", ha='center')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_summary.png", dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {output_dir}/benchmark_summary.png")

def main():
    parser = argparse.ArgumentParser(description="Run MOE overlap benchmarks and generate plots")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--experts_per_gpu", type=int, default=4, help="Number of experts per GPU")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per benchmark")
    parser.add_argument("--seq_lens", type=str, default="32,64,128,256", help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save results and plots")
    
    args = parser.parse_args()
    
    # Parse sequence lengths
    seq_lens = [int(seq_len) for seq_len in args.seq_lens.split(',')]
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save benchmark parameters
    with open(f"{output_dir}/benchmark_params.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Run benchmarks for each sequence length
    results = []
    for seq_len in seq_lens:
        result = run_benchmark_pair(
            args.num_gpus,
            args.experts_per_gpu,
            args.num_layers,
            args.batch_size,
            seq_len,
            args.iterations
        )
        results.append(result)
        
        # Save intermediate results
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Create and save plots
    create_plots(results, output_dir)
    
    print(f"\nAll benchmarks completed. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()