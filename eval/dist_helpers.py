import os
import torch.distributed as dist
import torch


def gather_objects_to_rank0(obj):
    if not dist_is_initialized():
        return [obj]

    rank = get_rank()
    world = get_world_size()

    gathered = [None] * world
    dist.all_gather_object(gathered, obj)

    # Only rank 0 returns the gathered payload
    return gathered if rank == 0 else None

def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()

def dist_setup(backend="nccl"):
    if dist_is_initialized(): 
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        # single process fallback
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

def get_rank():
    return dist.get_rank() if dist_is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist_is_initialized() else 1

def barrier():
    if dist_is_initialized(): dist.barrier()

def shard_list(x, rank, world_size):
    return [v for i, v in enumerate(x) if (i % world_size) == rank]

def rank_print(rank, *args):
    if rank == 0:
        print(*args)

def ddp_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return world_size, rank, local_rank

def ddp_init_if_needed():
    ws, rk, lr = ddp_env()
    if ws > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(lr)
    return ws, rk, lr