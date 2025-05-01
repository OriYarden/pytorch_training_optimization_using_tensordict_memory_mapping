import torch
import numpy as np
import inspect


def put_batch_on_device(batch, DEVICE='cpu'):
    # batch: [tuple] of Tensors.
    return [
        b.to(device=DEVICE) for b in batch
    ]


def stack_batches_onto_model_device(batches_from_all_devices, DEVICE='cpu'):
    # batches_from_all_devices: output(s) from (one or more) torch.utils.data.DataLoader(s).
    # DEVICE: gpu that has model gradients on it, this func puts batches_from_all_devices
    # on the same device as the model so that loss.backward() works.
    num_memmaps = len(batches_from_all_devices)
    num_items_in_a_memmap = len(batches_from_all_devices[-1])
    return put_batch_on_device(
        [
            torch.concatenate(
                [batches_from_all_devices[memmap_index][batch_item_index] for memmap_index in range(num_memmaps)],
                dim=0,
            ) for batch_item_index in range(num_items_in_a_memmap)
        ],
        DEVICE=DEVICE,
    )


def enumerate_loaders_in_parallel(loaders):
    # loaders: [list] of torch.utils.data.DataLoader(s).
    from tqdm import tqdm
    return tqdm(enumerate(zip(*loaders)), total=len(loaders[0]))


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
