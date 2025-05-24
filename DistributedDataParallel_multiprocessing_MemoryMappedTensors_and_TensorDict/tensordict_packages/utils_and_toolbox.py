import torch
from torch import multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import inspect
import os


def put_batch_on_device(batch, DEVICE='cpu'):
    # batch: [tuple] of Tensors.
    return [
        b.to(device=DEVICE) for b in batch
    ]


class AssertionMessage:
    def __init__(self, assert_message=None):
        self.assert_message = assert_message if assert_message is not None else ''

    def __str__(self):
        return self.assert_message


def get_var_names_returned_from_getitem_method(ds):
    """ Gets the names of variables in string that are returned in torch.utils.data.Dataset.__getitem__
    which will serve as the memory mapped tensordict.keys() used in tensordict_wrapper.py's
    dataset_to_tensordict wrapper function.

    Args:
    _____

        - ds: Pytorch Dataset instance [torch.utils.data.Dataset].

    Return:
    _______

        - return_vars: [List] of variable names [str] that are returned from torch.utils.data.Dataset.__getitem__


    >>> NOTE: The return statement in torch.utils.data.Dataset.__getitem__ must be a tuple with a trailing comma,
    regardless of how many items are being returned, and items cannot have any operations in the return statement.


    + EXAMPLES - correct:

        def __getitem__(self, index):
            ...
            ...
            ...
            return (
                some_var,
                some_other_var,
            )

        def __getitem__(self, index):
            ...
            ...
            ...
            return (
                some_var,
            )


    + EXAMPLES - incorrect:

        def __getitem__(self, index):
            ...
            ...
            ...
            return some_var

        def __getitem__(self, index):
            ...
            ...
            ...
            return (
                some_var.permute(2, 0, 1),
                some_other_var
            )
    """
    get_item_method_as_str = inspect.getsource(ds.__getitem__)
    return_vars = get_item_method_as_str[
        -get_item_method_as_str[::-1].find('(') + 1:-get_item_method_as_str[::-1].find(',') - 1
    ].split('\n')
    return [
        np.array(return_var.split()).astype(str)[0].replace(',', '') for return_var in return_vars
    ]


class ChunkedDatasetWrapper:
    def __init__(self, ds, chunk_start=0, chunk_end=0, is_last_chunk=False):
        self.ds = ds
        self.chunk_start = int(chunk_start)
        self.chunk_end = len(ds) if is_last_chunk else int(np.min((chunk_end, len(ds))))
        assert self.chunk_end - self.chunk_start > 0, '>>> Chunk indexes length is zero.'
        self.chunk_indexes = [
            index for index in range(self.chunk_start, self.chunk_end, 1)
        ]

    def __len__(self):
        return len(self.chunk_indexes)

    def __getitem__(self, index):
        chunk_index = self.chunk_indexes[index]
        return self.ds[chunk_index]


from .tensordict_wrapper import dataset_to_tensordict
def chunk_Dataset_to_fit_memmaps_onto_gpus(DS, MEMMAP_DEVICES=['cuda:0'], use_tensordict=False, force_number_of_chunks=0, **DS_kwargs):
    assertion_message = f'force_number_of_chunks = {force_number_of_chunks} requires {force_number_of_chunks} GPUs, but only {len(MEMMAP_DEVICES)} GPUs provided.'
    assert force_number_of_chunks <= len(MEMMAP_DEVICES), assertion_message
    ds = DS(
        use_tensordict=use_tensordict,
        **DS_kwargs,
    )
    list_of_tensordict_keys = get_var_names_returned_from_getitem_method(ds)
    chunked_ds_len = len(ds) if not force_number_of_chunks else len(ds) // force_number_of_chunks
    num_memmaps = 1 if not force_number_of_chunks else force_number_of_chunks

    ds_as_list_of_chunks = []
    chunk_len_has_been_found = False
    while not chunk_len_has_been_found:
        try:
            chunked_ds = ChunkedDatasetWrapper(
                ds=ds,
                chunk_end=chunked_ds_len,
            )
            if use_tensordict:
                print(f'\n>>> TRYING TO CHUNK DATASET ONTO {len(ds) // chunked_ds_len} GPU' + ['', 's'][(len(ds) // chunked_ds_len) > 1])
                chunked_ds = dataset_to_tensordict(
                    ds=chunked_ds,
                    DEVICE=MEMMAP_DEVICES[0],
                    force_list_of_tensordict_keys=list_of_tensordict_keys,
                )

            chunk_len_has_been_found = True
            ds_as_list_of_chunks.append(chunked_ds)
            for n in range(1, len(ds) // chunked_ds_len, 1):
                chunked_ds = ChunkedDatasetWrapper(
                    ds=ds,
                    chunk_start=chunked_ds_len*n,
                    chunk_end=chunked_ds_len*(n + 1),
                    is_last_chunk=(n == ((len(ds) // chunked_ds_len) - 1))
                )
                if use_tensordict:
                    chunked_ds = dataset_to_tensordict(
                        ds=chunked_ds,
                        DEVICE=MEMMAP_DEVICES[n],
                        force_list_of_tensordict_keys=list_of_tensordict_keys,
                    )
                ds_as_list_of_chunks.append(chunked_ds)
        except:
            num_memmaps += 1
            assert num_memmaps <= len(MEMMAP_DEVICES), '>>> REQUIRES MORE GPUs TO MEMORY MAP DATASET.'
            chunked_ds_len = len(ds) // num_memmaps
    return ds_as_list_of_chunks


def setup(rank, world_size, backend='nccl', master_addr='localhost', master_port='12355'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_multiprocessing(train_fn, Model, loaders, world_size, MEMMAP_DEVICES):
    print('\nSPAWNING PROCESSES\n')
    mp.spawn(train_fn,
             args=(world_size, Model, loaders, MEMMAP_DEVICES,),
             nprocs=world_size,
             join=True)
