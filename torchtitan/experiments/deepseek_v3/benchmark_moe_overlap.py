import time
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from model import DeepseekForCausalLM, MoE
from model_config import deepseek_config_registry
import argparse

def benchmark_moe_implementations(args):
    # Initialize distributed
    mesh = dist.init_device_mesh("cuda", (1, args.num_gpus), mesh_dim_names=("pp", "ep"))
    
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)  # Explicitly set device
    
    # Model configuration
    model_id = "deepseek-ai/DeepSeek-V2-Lite"
    model_args = deepseek_config_registry[model_id]
    
    # Adjust model size for testing
    model_args.num_hidden_layers = args.num_layers
    model_args.n_routed_experts = args.num_gpus * args.experts_per_gpu
    model_args.ep_size = args.num_gpus
    model_args.num_stages = 1
    model_args.stage_idx = 0
    
    # ======= CRITICAL CHANGE: We're going to run just ONE benchmark per script execution =======
    # This avoids device conflicts between the two benchmarks
    
    if rank == 0:
        print(f"Benchmarking MOE with:")
        print(f"  - {model_args.n_routed_experts} total experts across {model_args.ep_size} GPUs")
        print(f"  - {model_args.num_hidden_layers} layers")
        print(f"  - Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
        print(f"  - Mode: {'With Overlap' if args.use_overlap else 'Traditional (no overlap)'}")
    
    # Create synthetic data
    batch_size = args.batch_size
    seq_len = args.seq_len
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Create model and move to device
    with mesh:
        model = DeepseekForCausalLM(model_args)
    
    # If using overlap, enable symmetric memory
    if args.use_overlap:
        model.setup_symm_mem(torch.bfloat16, device)
    
    model.train()
    
    # Ensure input tensors are on the correct device
    x = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(3):
        output = model(x)
        if output.dim() == 3:  # [batch, seq, vocab]
            loss = loss_fn(output.reshape(-1, output.size(-1)), labels.reshape(-1))
        else:
            loss = loss_fn(output, labels)
        loss.backward()
        model.zero_grad()
    
    # Benchmark
    if rank == 0:
        print(f"\nBenchmarking {'WITH' if args.use_overlap else 'WITHOUT'} overlap...")
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(args.iterations):
        output = model(x)
        if output.dim() == 3:
            loss = loss_fn(output.reshape(-1, output.size(-1)), labels.reshape(-1))
        else:
            loss = loss_fn(output, labels)
        loss.backward()
        
        if rank == 0 and i == 0:
            print(f"Loss: {loss.item():.4f}")
        
        model.zero_grad()
    
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start) / args.iterations * 1000
    
    # Calculate throughput
    tokens_per_batch = batch_size * seq_len
    throughput = tokens_per_batch / (elapsed_time / 1000)
    
    # Gather all times to rank 0
    all_times = [0.0] * dist.get_world_size()
    dist.all_gather_object(all_times, elapsed_time)
    all_throughputs = [0.0] * dist.get_world_size()
    dist.all_gather_object(all_throughputs, throughput)
    
    # Report results from rank 0 only
    if rank == 0:
        avg_time = sum(all_times) / len(all_times)
        avg_throughput = sum(all_throughputs) / len(all_throughputs)
        
        # Print in a format that can be easily parsed by the plotting script
        prefix = "Comet (with overlap)" if args.use_overlap else "Vanilla (no overlap)"
        print(f"\n=== Performance Results ===")
        print(f"{prefix}: {avg_time:.2f} ms/iter")
        print(f"Tokens/sec {'Comet' if args.use_overlap else 'vanilla'}: {avg_throughput:.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MoE implementation with and without communication-computation overlap")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--experts_per_gpu", type=int, default=4, help="Number of experts per GPU")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to benchmark")
    parser.add_argument("--use_overlap", action="store_true", help="Use overlap implementation")
    args = parser.parse_args()
    
    benchmark_moe_implementations(args)