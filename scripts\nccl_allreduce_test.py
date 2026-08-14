import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    print(
        f"NCCL_ALLREDUCE_OK host={socket.gethostname()} rank={rank} "
        f"local_rank={local_rank} value={value.item()}",
        flush=True,
    )
    if value.item() != 10.0:
        raise RuntimeError(f"unexpected all_reduce result: {value.item()}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
