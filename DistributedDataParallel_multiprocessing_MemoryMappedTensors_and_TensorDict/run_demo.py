import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.parallel import DistributedDataParallel as DDP
from demo_dataloader import SomePyTorchDataset, get_loader
from demo_model import SomePyTorchModel
from tensordict_packages.collate_wrapper import Collate_Fn
from tensordict_packages.utils_and_toolbox import put_batch_on_device, setup, cleanup, run_multiprocessing, ChunkedDatasetWrapper, chunk_Dataset_to_fit_memmaps_onto_gpus
from tqdm import tqdm


def train_fn(rank, world_size, Model, loaders, MEMMAP_DEVICES, num_epochs=2):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    orig_model = Model().to(device=MEMMAP_DEVICES[rank])
    model = DDP(orig_model, device_ids=[rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        print(f'[EPOCH {epoch + 1} of {num_epochs}]')
        for i, batch in tqdm(enumerate(loaders[rank]), total=len(loaders[rank])):
            image, mask, mask_2d, some_binary_label, some_multi_labels = put_batch_on_device(batch, DEVICE=MEMMAP_DEVICES[rank], non_blocking=True)
            optimizer.zero_grad()
            prediction = model(image)
            loss = loss_fn(prediction, mask)
            loss.backward()
            optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


if __name__ == '__main__':
    # Here we use torch multiprocessing, torch.distributed, DistributedDataParallel, MemoryMappedTensors, and TensorDict--all at once.
    # each process runs on a GPU, each GPU gets a model replica and memory map of Dataset, all processes spawn in parallel.

    force_number_of_chunks = 2 # number of chunks = number of GPUs used = number of memory maps = number of processes running in parallel.
    batch_size = 8
    image_size = 256
    MEMMAP_DEVICES = [ # gpu(s) for the dataloader(s) memory mapped tensordict, DDP, distributed, and torch multiprocessing for model parallelism.
        f'cuda:{device_index}' for device_index in range(torch.cuda.device_count())
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

    for use_tensordict, print_statement in enumerate(['multiprocessing + torch.distributed + DDP', 'multiprocessing + torch.distributed + DDP + MemoryMappedTensors + TensorDict']):
        print(print_statement)
        # split torch.utils.data.Dataset into separate chunks (i.e. subsets).
        # to max out training speed, use the largest possible chunk size that can fit onto GPU(s).
        # e.g. if memmaps and model can fit on 2 GPUs then using 4 GPUs will be slower.
        ds_as_list_of_chunks = chunk_Dataset_to_fit_memmaps_onto_gpus(
            DS=SomePyTorchDataset,
            MEMMAP_DEVICES=MEMMAP_DEVICES,
            use_tensordict=use_tensordict,
            force_number_of_chunks=force_number_of_chunks, # <--- comment out to let func determine the number of chunks to use.
            **{
                'batch_size': batch_size,
                'image_size': image_size,
                'path_to_dataset_folder': path_to_dataset_folder,
            }
        )

        loaders = [ # loaders: list of torch.utils.data.Dataloader(s), each Dataloader has a Dataset chunk (subset of original Dataset).
            get_loader(
                ds=ds_as_list_of_chunks[i],
                batch_size=batch_size,
                use_tensordict=use_tensordict,
                #collate_fn=None, <--- can be None if not use_tensordict.
                collate_fn=Collate_Fn(
                    batch_size=batch_size,
                    DEVICE=MEMMAP_DEVICES[i],
                    #transform=None, # <--- can be None if use_tensordict and there are no probability based transforms.
                    transform=transform, # <--- if use_tensordict, can input probability based transforms (which cannot be in Dataset).
                ),
            ) for i in range(len(ds_as_list_of_chunks))
        ]

        run_multiprocessing(
            train_fn=train_fn,
            Model=SomePyTorchModel,
            loaders=loaders,
            MEMMAP_DEVICES=MEMMAP_DEVICES,
            world_size=len(loaders),
        )

