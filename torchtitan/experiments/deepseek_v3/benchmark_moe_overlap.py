import time
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from model import DeepseekForCausalLM, MoE
from model_config import deepseek_config_registry
import argparse

def benchmark_moe_implementations(args):
    # Initialize distributed
    mesh = dist.init_device_mesh("cuda", (1, args.num_gpus), mesh_dim_names=("pp", "ep"))
    
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    
    # Model configuration
    model_id = "deepseek-ai/DeepSeek-V2-Lite"
    model_args = deepseek_config_registry[model_id]
    
    # Adjust model size for testing
    model_args.num_hidden_layers = args.num_layers
    model_args.n_routed_experts = args.num_gpus * args.experts_per_gpu
    model_args.ep_size = args.num_gpus
    model_args.num_stages = 1
    model_args.stage_idx = 0
    
    if rank == 0:
        print(f"Benchmarking MOE with:")
        print(f"  - {model_args.n_routed_experts} total experts across {model_args.ep_size} GPUs")
        print(f"  - {model_args.num_hidden_layers} layers")
        print(f"  - Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    
    # Create model
    with device, mesh:
        model = DeepseekForCausalLM(model_args)
    
    # Enable symmetric memory for both implementations
    # (we'll toggle the shuffle method later)
    model.setup_symm_mem(torch.bfloat16, device)
    model.train()
    
    # Create optimizer (for completeness in training benchmark)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create synthetic data
    batch_size = args.batch_size
    seq_len = args.seq_len
    x = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(3):
        output = model(x)
        if output.dim() == 3:  # [batch, seq, vocab]
            loss = loss_fn(output.reshape(-1, output.size(-1)), labels.reshape(-1))
        else:
            loss = loss_fn(output, labels)
        loss.backward()
        optimizer.zero_grad()
    
    # Benchmark with overlap (symm_mem with streams)
    if rank == 0:
        print("\nBenchmarking WITH Comet overlap...")
    
    # Set the shuffle method to symm_mem 
    for layer in model.model.layers.values():
        if hasattr(layer.mlp, 'shuffle_method'):
            layer.mlp.shuffle_method = "symm_mem"
    
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
        
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    overlap_time = (time.time() - start) / args.iterations * 1000
    
    # Benchmark without overlap (traditional all-to-all)
    if rank == 0:
        print("\nBenchmarking WITHOUT overlap...")
    
    # Change the shuffle method
    for layer in model.model.layers.values():
        if hasattr(layer.mlp, 'shuffle_method'):
            layer.mlp.shuffle_method = "torch_all_to_all"
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(args.iterations):
        output = model(x)
        if output.dim() == 3:
            loss = loss_fn(output.reshape(-1, output.size(-1)), labels.reshape(-1))
        else:
            loss = loss_fn(output, labels)
        loss.backward()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    vanilla_time = (time.time() - start) / args.iterations * 1000
    
    # Report results
    tokens_per_batch = batch_size * seq_len
    if rank == 0:
        print(f"\n=== Performance Results ===")
        print(f"Vanilla (no overlap): {vanilla_time:.2f} ms/iter")
        print(f"Comet (with overlap): {overlap_time:.2f} ms/iter")
        print(f"Speedup: {vanilla_time/overlap_time:.2f}x")
        print(f"Improvement: {((vanilla_time - overlap_time) / vanilla_time * 100):.1f}%")
        print(f"Tokens/sec vanilla: {tokens_per_batch / (vanilla_time / 1000):.0f}")
        print(f"Tokens/sec Comet: {tokens_per_batch / (overlap_time / 1000):.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MoE implementation with and without communication-computation overlap")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--experts_per_gpu", type=int, default=4, help="Number of experts per GPU")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to benchmark")
    args = parser.parse_args()
    
    benchmark_moe_implementations(args)
    dist.destroy_process_group()