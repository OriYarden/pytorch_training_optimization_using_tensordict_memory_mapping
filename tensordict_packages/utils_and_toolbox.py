import torch
import numpy as np
import inspect


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
