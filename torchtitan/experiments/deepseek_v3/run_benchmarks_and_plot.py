import os
import json
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_benchmark(num_gpus, experts_per_gpu, num_layers, batch_size, seq_len, iterations):
    """Run the benchmark script with specified parameters"""
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
    
    print(f"\n{'='*80}")
    print(f"Running benchmark with sequence length: {seq_len}")
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
    with open(f"benchmark_output_seq_{seq_len}.txt", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
    
    # Parse results - look in the whole output
    results = {"seq_len": seq_len}
    output = result.stdout
    
    # Try to find the result lines
    vanilla_line = [line for line in output.split('\n') if "Vanilla (no overlap):" in line]
    if vanilla_line:
        try:
            results["vanilla_time"] = float(vanilla_line[0].split(":")[1].strip().split()[0])
        except (IndexError, ValueError) as e:
            print(f"Error parsing vanilla time: {e}")
    
    overlap_line = [line for line in output.split('\n') if "Comet (with overlap):" in line]
    if overlap_line:
        try:
            results["overlap_time"] = float(overlap_line[0].split(":")[1].strip().split()[0])
        except (IndexError, ValueError) as e:
            print(f"Error parsing overlap time: {e}")
    
    speedup_line = [line for line in output.split('\n') if "Speedup:" in line]
    if speedup_line:
        try:
            results["speedup"] = float(speedup_line[0].split(":")[1].strip().split("x")[0])
        except (IndexError, ValueError) as e:
            print(f"Error parsing speedup: {e}")
    
    improvement_line = [line for line in output.split('\n') if "Improvement:" in line]
    if improvement_line:
        try:
            results["improvement"] = float(improvement_line[0].split(":")[1].strip().split("%")[0])
        except (IndexError, ValueError) as e:
            print(f"Error parsing improvement: {e}")
    
    vanilla_tokens_line = [line for line in output.split('\n') if "Tokens/sec vanilla:" in line]
    if vanilla_tokens_line:
        try:
            results["vanilla_tokens_per_sec"] = float(vanilla_tokens_line[0].split(":")[1].strip())
        except (IndexError, ValueError) as e:
            print(f"Error parsing vanilla tokens/sec: {e}")
    
    overlap_tokens_line = [line for line in output.split('\n') if "Tokens/sec Comet:" in line]
    if overlap_tokens_line:
        try:
            results["overlap_tokens_per_sec"] = float(overlap_tokens_line[0].split(":")[1].strip())
        except (IndexError, ValueError) as e:
            print(f"Error parsing overlap tokens/sec: {e}")
    
    if not all(k in results for k in ["vanilla_time", "overlap_time"]):
        # If we can't find the results, we need to make sure our benchmark script is correctly printing them
        print(f"WARNING: Could not parse complete results for seq_len={seq_len}")
        print("Please check if the benchmark script is printing the results in the expected format:")
        print("  - 'Vanilla (no overlap): X.XX ms/iter'")
        print("  - 'Comet (with overlap): X.XX ms/iter'")
        print("  - 'Speedup: X.XXx'")
        print("  - 'Improvement: XX.X%'")
        print("  - 'Tokens/sec vanilla: XXXX'")
        print("  - 'Tokens/sec Comet: XXXX'")
        
        # Use placeholder values to avoid breaking the plotting
        if "vanilla_time" not in results:
            results["vanilla_time"] = 0
        if "overlap_time" not in results:
            results["overlap_time"] = 0
        if "speedup" not in results:
            results["speedup"] = 1.0
        if "improvement" not in results:
            results["improvement"] = 0.0
        if "vanilla_tokens_per_sec" not in results:
            results["vanilla_tokens_per_sec"] = 0
        if "overlap_tokens_per_sec" not in results:
            results["overlap_tokens_per_sec"] = 0
    
    return results

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
        plt.text(i - bar_width/2, v + 1, f"{v:.1f}", ha='center')
    for i, v in enumerate(overlap_times):
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
    for i, v in enumerate(vanilla_throughput):
        if v > 0:
            plt.text(i - bar_width/2, v + max(filter(lambda x: x > 0, vanilla_throughput))*0.02, f"{int(v)}", ha='center')
    for i, v in enumerate(overlap_throughput):
        if v > 0:
            plt.text(i + bar_width/2, v + max(filter(lambda x: x > 0, overlap_throughput))*0.02, f"{int(v)}", ha='center')
    
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
    relative_throughput = [o/v if v > 0 else 1.0 for o, v in zip(overlap_throughput, vanilla_throughput)]
    plt.bar(x, relative_throughput, width=0.6, color='purple')
    plt.xlabel('Sequence Length')
    plt.ylabel('Relative Throughput')
    plt.title('Throughput Improvement')
    plt.xticks(x, seq_lengths_labels)
    for i, v in enumerate(relative_throughput):
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
        result = run_benchmark(
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