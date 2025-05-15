import torch
import torch.distributed as dist
import os
import sys

# Add the deepseek directory directly to avoid import chains
sys.path.insert(0, '/n/netscratch/idreos_lab/Lab/pseth/dv2/torchtitan/torchtitan/experiments/deepseek_v3')

# Initialize distributed environment
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

dist.init_process_group(backend='nccl', world_size=1, rank=0)

# Now import just what we need for DeepSeek
from model_config import ModelArgs

# Create minimal config
config = ModelArgs(
    hidden_size=2048,
    moe_intermediate_size=1024,
    n_routed_experts=4,
    ep_size=1,
    num_experts_per_tok=2,
    max_seq_len=512,
    vocab_size=32000,
    num_hidden_layers=1,
    n_shared_experts=1,
)

# Import MoE directly
from model import MoE

print("Creating MoE layer...")
moe = MoE(config)

# Set process group manually
moe.ep_group = dist.group.WORLD

# Test forward pass
x = torch.randn(16, config.hidden_size, device='cuda')
print(f"Input shape: {x.shape}")

output = moe(x)
print(f"Output shape: {output.shape}")

# Check if Comet implementation is there
print(f"\nComet Implementation Check:")
print(f"MoE has copy_stream: {hasattr(MoE, 'copy_stream')}")
print(f"MoE has comp_stream: {hasattr(MoE, 'comp_stream')}")
print(f"MoE has moe_on_device method: {hasattr(moe, 'moe_on_device')}")

dist.destroy_process_group()
print("\nTest completed successfully!")
