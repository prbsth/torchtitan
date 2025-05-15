import os
import torch
import torch.distributed as dist

# Set backend explicitly
os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'nccl'

# Bypass the localhost issue by initializing directly
dist.init_process_group(
    backend='nccl', 
    init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
    world_size=4,
    rank=int(os.environ.get('RANK', 0))
)

# Now run our test
from train import run_full_model
from model_config import deepseek_config_registry

config = deepseek_config_registry["deepseek-ai/DeepSeek-V2-Lite"]
config.num_hidden_layers = 2

mesh = dist.init_device_mesh("cuda", (1, 4, 1), mesh_dim_names=("pp", "ep", "fsdp"))
run_full_model(mesh)
