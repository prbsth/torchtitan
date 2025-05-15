import torch
import torch.distributed as dist
from train import run_full_model

# Quick test with tiny model
from model_config import deepseek_config_registry
config = deepseek_config_registry["deepseek-ai/DeepSeek-V2-Lite"]
config.num_hidden_layers = 2  # Just 2 layers to test quickly

mesh = dist.init_device_mesh("cuda", (1, 4, 1), mesh_dim_names=("pp", "ep", "fsdp"))
run_full_model(mesh)
