import os
import pytest
import time

import torch
from torchdata.datapipes.iter import IterDataPipe

class FooLister(IterDataPipe):
    def __iter__(self):
        for x in range(10):
            yield x

def training(rank, world_size, late_rank):
    pg = torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    # simulate data loading (when rank 0 arrives last test fails)
    time.sleep(6 if rank == late_rank else 0)
    
    dp = FooLister()
    dp = dp.shuffle().sharding_filter()

    dl = torch.utils.data.DataLoader(dp, num_workers=2)
    item = next(iter(dl))

@pytest.mark.parametrize(["world_size", "late_rank"], [(2,0), (8,0), (2,1), (8,7)])
def test_web_datapipe(world_size, late_rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.multiprocessing.spawn(
        training,
        args=(world_size, late_rank),
        nprocs=world_size,
        join=True,
    )