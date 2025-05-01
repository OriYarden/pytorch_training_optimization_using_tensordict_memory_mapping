from tensordict import TensorDict, MemoryMappedTensor
import torch
from .utils_and_toolbox import get_var_names_returned_from_getitem_method


def dataset_to_tensordict(ds, DEVICE='cpu', list_of_dtypes=None):
    """ This wrapper converts (wraps) a Pytorch Dataset to a memory mapped TensorDict which can be inputted
    into a Pytorch DataLoader the same way it would be if it were a typical Pytorch Dataset.

    This wrapper handles the keys, shapes, and dtypes for the TensorDict. The list_of_dtypes is an optional arg
    if one wants to set the dtypes.

    Args:
    _____

        - ds: Pytorch Dataset [torch.utils.data.Dataset].
        - DEVICE: nvidia gpu device (e.g. 'cpu', 'cuda:0').
        - list_of_dtypes: optional [List] of torch.dtypes that will be applied in the same order to match the
            items in the torch.utils.data.Dataset.__getitem__ return statement.
            Otherwise, default dtypes are torch.in64 for 'labels' and torch.float32 for 'images'.

    Return:
    _______

        - Pytorch tensordict.TensorDict.MemoryMappedTensor(Dataset), equivalent to a typical Pytorch torch.utils.data.Dataset
            and can be inputted into the Pytorch torch.utils.data.DataLoader the same way as torch.utils.data.Dataset
    """
    print('\nTensorDict Wrapper  ' +  ' '*len(str(len(ds))) + '|' + '\n' + '--------------------' + '-'*len(str(len(ds))) + '|')
    list_of_tensordict_keys = get_var_names_returned_from_getitem_method(ds)
    print('=>    [ MAPPING ]   ' + ' '*len(str(len(ds))) + '|')
    ds_as_a_tensordict = TensorDict({
            key: (
                MemoryMappedTensor.empty(
                    (len(ds),),
                    dtype=torch.int64 if list_of_dtypes is None else list_of_dtypes[i],
                )
            ) if len(ds[0][i].squeeze().shape) == 1 and ds[0][i].squeeze().shape[0] == 1 else (
                MemoryMappedTensor.empty(
                    (len(ds), *ds[0][i].shape),
                    dtype=[torch.float32, torch.int64][len(ds[0][i].squeeze().shape) == 1] if list_of_dtypes is None else list_of_dtypes[i],
                )
            ) for i, key in enumerate(list_of_tensordict_keys)
        },
        batch_size=[len(ds)],
        device=DEVICE, # maps onto device.
    )
    print('=>    [ PINNING ]   ' + ' '*len(str(len(ds))) + '|')
    ds_as_a_tensordict.memmap_() # pins memory map here.
    print('=>    [ LOADING ]   ' + ' '*len(str(len(ds))) + '|')
    from tqdm import tqdm
    for i, dataset_items in tqdm(enumerate(ds)): # loads onto (mapping) device.
        ds_as_a_tensordict[i] = TensorDict({
            list_of_tensordict_keys[key_index]: dataset_items[key_index] for key_index in range(len(list_of_tensordict_keys))
        }, batch_size=[])
    print('---------------------' + '-'*len(str(len(ds))) + '\n')
    return ds_as_a_tensordict
