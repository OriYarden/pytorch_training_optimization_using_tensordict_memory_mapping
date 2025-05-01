import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from demo_dataloader import get_loader
from tensordict_packages.collate_wrapper import Collate_Fn
from tensordict_packages.utils_and_toolbox import enumerate_loaders_in_parallel, stack_batches_onto_model_device


batch_size = 8
image_size = 256
DEVICE = 'cuda:0' # gpu for model and batch gradients, operations, etc.
MEMMAP_DEVICES = [ # gpu(s) for the dataloader(s) memory mapped tensordict, one gpu is equivalent to the base version in this repo.
    f'cuda:{device_index}' for device_index in range(torch.cuda.device_count()) if f'cuda:{device_index}' != DEVICE
]
path_to_dataset_folder = None
transform = A.Compose(
    [
        # if use_tensordict, probability based transforms must be defined here instead of in Dataset, and are inputted into Collate_Fn.
        A.Rotate(limit=0.5, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(), # <--- must include ToTensorV2().
    ], is_check_shapes=False,
    #additional_targets={'image': 'image', 'mask': 'mask'} # can leave as commented out since Collate_Fn handles this.
)


for use_tensordict, type_inputted_into_torch_DataLoader in enumerate(['torch.utils.data.Dataset <--- baseline', 'tensordict.TensorDict.MemoryMappedTensor <--- what the tensordict_packages wraps Dataset with']):
    loaders = [
        get_loader(
            batch_size=batch_size,
            image_size=image_size,
            DEVICE=device_to_memmap,
            path_to_dataset_folder=path_to_dataset_folder,
            use_tensordict=use_tensordict,
            #collate_fn=None, <--- can be None if not use_tensordict.
            collate_fn=Collate_Fn(
                batch_size=batch_size,
                DEVICE=device_to_memmap,
                #transform=None, # <--- can be None if use_tensordict and there are no probability based transforms.
                transform=transform, # <--- if use_tensordict, can input probability based transforms (which cannot be in Dataset).
            ),
        ) for device_to_memmap in MEMMAP_DEVICES
    ]

    num_epochs = [1, 3][use_tensordict]
    for epoch in range(num_epochs):
        print(f'\n>> [RUNNING MOCH EPOCH {epoch + 1} of {num_epochs}] {type_inputted_into_torch_DataLoader}\n')
        for i, super_batch in enumerate_loaders_in_parallel(loaders):
            # super_batch is all batches from all memory mapped gpus.
            # batch dim of super_batch is batch_size x number of memory mapped gpus.
            #
            # var names returned from Dataset.__getitem__ and Albumentations.Compose.addional_targets.keys() must all match if use_tensordict.
            # var names here below do not matter for tensordict memory mapping.
            image, mask, mask_2d, some_binary_label, some_multi_labels = stack_batches_onto_model_device(
                batches_from_all_devices=super_batch,
                DEVICE=DEVICE,
            )
            # prediction = model(image)
            # ...
            # ...
            # ...

    print('\n')
