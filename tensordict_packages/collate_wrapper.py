import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils_and_toolbox import AssertionMessage


class Collate_Fn:
    """ This class acts as a callable wrapper so that items from the TensorDict-DataLoader are stacked
    contiguously allowing those items to be augmented (via Albumentations) without memory mapping issues.

    Args:
    _____

        - DEVICE: nvidia gpu device (e.g. 'cpu', 'cuda:0').
        - transform: optional parameter. Input probability-based Albumentation's augmentations.
            The default option is None; it will use a dummy transform which has no augmentations.

            + transform input EXAMPLE:
            --------------------------

                transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        ToTensorV2(),
                    ], is_check_shapes=False,
                )

    Return:
    _______

        - A callable collate_fn wrapper; which is then inputted into the collate_fn parameter in torch.utils.data.DataLoader.


    -------------------------------------------------------------------------------------------------------------------------
    Probability-based augmentations (e.g. A.HorizontalFlip(p=0.5)) cannot be mapped to memory (via the
    dataset_to_tensordict wrapper) because those augmentations are pinned and therefore no longer based on probability.

    In order to use probability-based augmentations (e.g. A.RandomBrightnessContrast(p=0.5)), they must be applied
    here in the Collate_Fn __call__ method, not in the Pytorch torch.utils.data.Dataset.__getitem__ method:

        + Albumentations that do NOT need to be in Collate_Fn (i.e. they can remain in Pytorch Dataset's __getitem__ method):
        ------------------------------------------------------
            - A.Normalize()
            - A.Resize()
            - etc. (anything without probability)

        + Albumentations that NEED to be in Collate_Fn:
        ------------------------------------------------------
            - A.HorizontalFlip()
            - A.VerticalFlip()
            - etc. (anything that has 'p' for probability as a parameter)
    """
    def __init__(self, batch_size, DEVICE='cpu', transform=None):
        self.batch_size = batch_size
        self.DEVICE = DEVICE
        self.transform = transform

    @staticmethod
    def assertion_message(batch_keys, transform_keys):
        return AssertionMessage(
            f'''
            Albumentations.Compose.additional_targets.keys() == list_of_tensordict_keys must be True.

            Please re-name keys:
            ____________________

            >>> transform = A.Compose(
            >>>     [
            >>>         ...,
            >>>     ], ...,
            >>>     additional_targets={"{"}
            >>>     ------------------
            >>>         {": ... , ".join(list(transform_keys)[:len(batch_keys)])}: ...
            >>>     {"}"}
            >>> )

            to:
            ___

            >>> transform = A.Compose(
            >>>     [
            >>>         ...,
            >>>     ], ...,
            >>>     additional_targets={"{"}
            >>>     ------------------
            >>>         {": ... , ".join(batch_keys)}: ...
            >>>     {"}"}
            >>> )
            '''
        )

    @staticmethod
    def return_a_do_nothing_transform(additional_targets):
        # dummy transform.
        # transform without augmentations.
        return A.Compose(
            [
                ToTensorV2(),
            ], is_check_shapes=False,
            additional_targets=additional_targets
        )

    @staticmethod
    def transpose_if_necessary(x):
        # CHW -> HWC
        return x.transpose(1, 2, 0) if len(x.shape) == 3 else x

    def return_album_type(self, x):
        if len(x.shape) <= 2 and self.batch_size in x.shape:
            # for 1-dimensional tensors (i.e. 'label' or 'class' instead of 'image' or 'mask')
            # '' is used because it is skipped over when applying augmentations.
            # no augmentations are applied to 1-dimensional tensors in this Collate_Fn wrapper class.
            return ''
        if len(x.shape) == 3:
            return 'mask'
        return 'image'

    def yield_stack_wrapper(self, **kwargs):
        # stacks one item (i.e. batch[key]) over the batch dimension.
        def yield_stack(batch_as_list, key_index, batch_size):
            for i in range(key_index, len(batch_as_list), int(len(batch_as_list) / batch_size)):
                yield batch_as_list[i]

        return torch.stack(
            list(yield_stack(**kwargs)),
            dim=0,
        )

    def apply_albumentations(self, batch):
        # apply augmentations per batch (i.e. apply to items within-batch-dimension instead of same item over batch-dimension).
        # otherwise augmentations won't be applied to the items contiguously (e.g. random flips would not be the same for corresponding image-mask pairs).
        # and then append augmented items over batch-dimension.
        # and then return stacked (contiguous) items over batch-dimension.
        batch_as_list = [
            batch[key][batch_i, ...] if not self.transform.additional_targets[key] else self.transpose_if_necessary(
                batch[key][batch_i, ...].detach().cpu().numpy() # augmentations only for more than 1-dimensional tensors.
            ) for batch_i in range(batch.shape[0]) for key in batch.keys()
        ]
        augmented_batch_as_list = []
        for batch_i in range(0, len(batch_as_list), len(batch.keys())):
            augs = self.transform(
                **{
                    key: batch_as_list[i] for i, key in zip(
                        range(batch_i, batch_i + len(batch.keys()), 1),
                        batch.keys()
                    ) if self.transform.additional_targets[key] # if-statement to skip augmentations for 1-dimensional tensors.
                }
            )
            augmented_batch_as_list += [
                augs[key] if self.transform.additional_targets[key] else batch[key][ # apply augmentations to only 'image' and 'mask' targets.
                    batch_i // len(batch.keys()) # else don't apply augmentations (for 1-dimensional tensors).
                ] for key in batch.keys()
            ]
        return [
            self.yield_stack_wrapper(
                **{
                    'batch_as_list': augmented_batch_as_list,
                    'key_index': key_index,
                    'batch_size': int(len(augmented_batch_as_list) / len(batch.keys()))
                }
            ) for key_index in range(len(batch.keys()))
        ]

    def __call__(self, batch):
        if self.transform is None:
            self.transform = self.return_a_do_nothing_transform(
                additional_targets={
                    key: self.return_album_type(batch[key]) for key in batch.keys()
                }
            )
        if batch.keys() != self.transform.additional_targets.keys():
            self.transform.add_targets({
                key: self.return_album_type(batch[key]) for key in batch.keys()
            })
        assert batch.keys() == self.transform.additional_targets.keys(), self.assertion_message(batch.keys(), self.transform.additional_targets.keys())
        return self.apply_albumentations(batch)
