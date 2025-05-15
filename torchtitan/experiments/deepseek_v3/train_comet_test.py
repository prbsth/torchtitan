import time
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

def benchmark_training_overlap():
    # Use 1 pipeline stage, 4 expert parallel
    mesh = dist.init_device_mesh("cuda", (1, 4, 1), mesh_dim_names=("pp", "ep", "fsdp"))
    
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    
    # Model configuration
    model_id = "deepseek-ai/DeepSeek-V2-Lite"
    model_args = deepseek_config_registry[model_id]
    
    # Reduce model size for testing
    model_args.num_hidden_layers = 4  # Just 4 layers for testing
    model_args.n_routed_experts = 16  # 4 experts per GPU
    model_args.ep_size = 4
    model_args.num_stages = 1
    model_args.stage_idx = 0
    
    if rank == 0:
        print(f"Training benchmark with {model_args.n_routed_experts} experts on {model_args.ep_size} GPUs")
        print(f"Layers: {model_args.num_hidden_layers}")
    
    # Create model
    with device, mesh:
        model = DeepseekForCausalLM(model_args)
    
    # Enable symmetric memory for overlap
    model.setup_symm_mem(torch.bfloat16, device)
    model.train()
    
    # Apply FSDP
    fsdp_mesh = mesh["fsdp"]
    hsdp_mesh = mesh["ep", "fsdp"]
    
    for layer in model.model.layers.values():
        if hasattr(layer.mlp, "experts"):
            for expert in layer.mlp.experts.values():
                fully_shard(expert, mesh=fsdp_mesh, reshard_after_forward=False)
        fully_shard(layer, mesh=hsdp_mesh, reshard_after_forward=False)
    
    fully_shard(model, mesh=hsdp_mesh, reshard_after_forward=False)
    
    # Training parameters
    batch_size = 4
    seq_len = 512
    num_iterations = 10
    
    # Create synthetic data
    x = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model_args.vocab_size, 
                          (batch_size, seq_len), device=device)
    
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
        model.zero_grad()
    
    # Benchmark with overlap
    if rank == 0:
        print("\nBenchmarking WITH Comet overlap...")
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(num_iterations):
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
    overlap_time = (time.time() - start) / num_iterations * 1000
    
    # Benchmark without overlap (change shuffle method)
    if rank == 0:
        print("\nBenchmarking WITHOUT overlap...")
    
    # Disable overlap
    for layer in model.model.layers.values():
        if hasattr(layer.mlp, 'shuffle_method'):
            layer.mlp.shuffle_method = "torch_all_to_all"
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(num_iterations):
        output = model(x)
        if output.dim() == 3:
            loss = loss_fn(output.reshape(-1, output.size(-1)), labels.reshape(-1))
        else:
            loss = loss_fn(output, labels)
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    vanilla_time = (time.time() - start) / num_iterations * 1000
    
    # Report results
    if rank == 0:
        print(f"\n=== Training Performance Results ===")
        print(f"Vanilla (no overlap): {vanilla_time:.2f} ms/iter")
        print(f"Comet (with overlap): {overlap_time:.2f} ms/iter")
        print(f"Speedup: {vanilla_time/overlap_time:.2f}x")
        print(f"Improvement: {((vanilla_time - overlap_time) / vanilla_time * 100):.1f}%")
        print(f"Tokens/sec vanilla: {batch_size * seq_len / (vanilla_time / 1000):.0f}")
        print(f"Tokens/sec Comet: {batch_size * seq_len / (overlap_time / 1000):.0f}")

if __name__ == "__main__":
    benchmark_training_overlap()
    dist.destroy_process_group()