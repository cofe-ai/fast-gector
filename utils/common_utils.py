from contextlib import contextmanager
import torch
from utils.mismatched_utils import *
from src.dataset import MyCollate, Seq2EditDataset
from torch.utils.data import DataLoader
import json
import torch.multiprocessing as mp

def read_config(path):
    with open(path, "r", encoding="utf8") as fr:
        config = json.load(fr)
    return config


def init_sampler(dataset, shuffle: bool, is_distributed: bool):
    if is_distributed:
        sampler = torch.utils.data.DistributedSampler(dataset=dataset,
                                     shuffle=shuffle,
                                     drop_last=True)
    else:
        sampler = None
    return sampler


def init_dataloader(subset,
                    data_path,
                    num_workers,
                    use_cache,
                    tokenizer,
                    vocab,
                    input_pad_id,
                    detect_pad_id,
                    correct_pad_id,
                    max_len,
                    batch_size,
                    tag_strategy,
                    skip_complex,
                    skip_correct=0,
                    tp_prob=1,
                    tn_prob=1):

    my_collate_fn = MyCollate(input_pad_id, detect_pad_id, correct_pad_id)

    sub_dataset = Seq2EditDataset(data_path,
                                  use_cache,
                                  tokenizer,
                                  vocab,
                                  max_len,
                                  tag_strategy,
                                  skip_complex,
                                  skip_correct,
                                  tp_prob,
                                  tn_prob)

    if subset == "train":
        shuffle = True
    else:
        shuffle = False
    
    is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

    sampler = init_sampler(dataset=sub_dataset,
                        shuffle=shuffle,
                        is_distributed=is_distributed)
    
    if is_distributed:
        # sampler option is mutually exclusive with shuffle
        shuffle = None
    
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # kwargs = dict()
    # if (is_distributed and num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    
    # set pin_memory to True seems to cause more GPU memory are cached, 
    # which may lead to OOM.
    # Thus, we disable it as a temporary solution
    data_loader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=my_collate_fn,
        num_workers=num_workers,
        sampler=sampler,
        # **kwargs
    )
    return data_loader




@contextmanager
def torch_distributed_master_process_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()
