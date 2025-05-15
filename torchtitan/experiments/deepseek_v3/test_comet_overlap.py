#!/usr/bin/env python3
import os
import sys
import time
import torch
import torch.distributed as dist

# Verify we're using the right torchtitan
import torchtitan
print(f"Using torchtitan from: {torchtitan.__file__}")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import create_model, create_dist_config
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

def test_comet_overlap():
    model_id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    mesh_shape = (1, 4)  # 1 pipeline, 4 expert parallel
    
    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("pp", "ep"))
    rank = dist.get_rank()
    
    if rank == 0:
        print(f"Testing Comet overlap implementation on {torch.cuda.device_count()} GPUs")
        print(f"Model: {model_id}")
    
    dist_config = create_dist_config(mesh)
    model, _ = create_model(dist_config)
    
    # Verify we have the overlap implementation
    moe_layer = model.model.layers["0"].mlp
    if hasattr(moe_layer, 'moe_on_device'):
        print(f"Rank {rank}: Comet overlap implementation detected ✓")
    else:
        print(f"Rank {rank}: WARNING - Using vanilla implementation")
    
    # Test parameters
    batch_size = 4
    seq_len = 512
    num_iterations = 10
    
    # Create test input
    x = torch.randint(0, 1000, (batch_size, seq_len), device=dist_config.device)
    
    # Warmup
    for _ in range(3):
        _ = model(x)
        torch.cuda.synchronize()
    
    # Test vanilla (no overlap)
    for layer in model.model.layers.values():
        if hasattr(layer.mlp, 'shuffle_method'):
            layer.mlp.shuffle_method = "torch_all_to_all"
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = model(x)
    torch.cuda.synchronize()
    vanilla_time = (time.time() - start) / num_iterations * 1000
    
    # Test with overlap
    for layer in model.model.layers.values():
        if hasattr(layer.mlp, 'shuffle_method'):
            layer.mlp.shuffle_method = "symm_mem"
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = model(x)
    torch.cuda.synchronize()
    overlap_time = (time.time() - start) / num_iterations * 1000
    
    if rank == 0:
        print(f"\nPerformance Results:")
        print(f"Vanilla (no overlap): {vanilla_time:.2f} ms")
        print(f"Comet (with overlap): {overlap_time:.2f} ms")
        print(f"Speedup: {vanilla_time/overlap_time:.2f}x")
        print(f"Improvement: {((vanilla_time - overlap_time) / vanilla_time * 100):.1f}%")

if __name__ == "__main__":
    test_comet_overlap()
    dist.destroy_process_group()