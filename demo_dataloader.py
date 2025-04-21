import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os


class SomePyTorchDataset(Dataset):
    def __init__(
            self,
            batch_size=8,
            image_size=512,
            path_to_dataset_folder=None,
            use_tensordict=False,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        # How the final dataset list that __getitem__ indexes is constructed does not matter.
        # Example for demo:
        self.path_to_dataset_folder = path_to_dataset_folder if path_to_dataset_folder is not None else (str(os.getcwd()) + f'/{self.__class__.__name__}/')
        self.create_folders(self.path_to_dataset_folder)
        self.transform = self.get_transforms(image_size, use_tensordict)
        self.images_filenames = [
            fn for fn in os.listdir(self.path_to_dataset_folder) if fn.endswith('.png')
        ] if os.path.exists(self.path_to_dataset_folder) else []
        if not len(self.images_filenames):
            self.create_demo_dataset()
            self.images_filenames = [
                fn for fn in os.listdir(self.path_to_dataset_folder) if fn.endswith('.png')
            ]
        self.dataset_as_tuples = [
            (
                self.path_to_dataset_folder + fn, # image full path and filename.
                self.path_to_dataset_folder + fn.replace('image', 'mask'), # corresponding mask full path and filename.
                (torch.rand(1) > 0.5).long(), # binary label 1-dim example.
                (torch.rand(5) > 0.5).long(), # multi labels 1-dim example.
            ) for fn in self.images_filenames
        ]

    def __len__(self):
        return len(self.dataset_as_tuples)

    @staticmethod
    def create_folders(folders):
        # folders: full path and folder name, or list of full paths and folder names.
        for folder in [folders] if not isinstance(folders, list) else folders:
            os.makedirs(folder, exist_ok=True)

    def create_demo_dataset(self, num_images=1001):
        original_image_shape = (int(self.image_size*2), int(self.image_size*2), 3)
        original_mask_shape = (int(self.image_size*2), int(self.image_size*2), 1)
        print(f'\n>>> CREATING DEMO DATASET: {num_images} images and {num_images} masks\n')
        from tqdm import tqdm
        for image_num in tqdm(range(num_images), total=num_images):
            image = np.random.random(original_image_shape)
            image = (image*255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(self.path_to_dataset_folder + f'image_{image_num}.png', format='PNG')
            mask = np.random.random(original_mask_shape)
            mask = (mask > 0.5).astype(float)
            mask = np.concatenate([mask for _ in range(3)], axis=2)
            mask = (mask*255).astype(np.uint8)
            mask = Image.fromarray(mask)
            mask.save(self.path_to_dataset_folder + f'mask_{image_num}.png', format='PNG')

    @staticmethod
    def get_transforms(image_size, use_tensordict):
        # returns 3 transforms: norm, main, and to_tensor.
        # norm: normalizes image and mask.
        # main: resize (and rotate, flip, etc. if not use_tensordict) for image and mask.
        # to_tensor: for image and mask.
        if use_tensordict:
            # when using tensordict memory mapping, probabilities should not be included in Dataset.
            # probability based transforms must be moved from Dataset into Collate_Fn.
            # all non-probability based transforms should be here in Dataset.
            return A.Compose(
                    [
                        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255),
                        A.Resize(height=image_size, width=image_size),
                        # move probability based transforms to Collate_Fn if use_tensordict.
                        ToTensorV2(),
                    ], is_check_shapes=False,
                    additional_targets={'image': 'image', 'mask': 'mask'}
                )
        # when not using tensordict memory mapping, all transforms should be included in Dataset.
        return A.Compose(
                [
                    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255),
                    A.Resize(height=image_size, width=image_size),
                    # if not use_tensordict, probability based transforms go here.
                    A.Rotate(limit=0.5, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    ToTensorV2(),
                ], is_check_shapes=False,
                additional_targets={'image': 'image', 'mask': 'mask'}
            )

    @staticmethod
    def load_image(fp_and_fn):
        # fp_and_fn: full path and filename
        image = Image.open(fp_and_fn)
        image = np.asarray(image).astype(np.float32)
        return image[:, :, :3]

    @staticmethod
    def load_mask(fp_and_fn):
        # fp_and_fn: full path and filename
        mask = Image.open(fp_and_fn)
        mask = np.asarray(mask).astype(np.float32)
        return mask[:, :, :3]

    def __getitem__(self, index):
        image_fp_and_fn, mask_fp_and_fn, binary_label, multi_label = self.dataset_as_tuples[index]

        image = self.load_image(image_fp_and_fn)
        mask = self.load_mask(mask_fp_and_fn)

        augs = self.transform(image=image, mask=mask)
        image, mask = augs['image'], augs['mask']

        mask = mask.permute(2, 0, 1) # <--- 3d dim mask.
        mask_2d = mask[0, :, :]#.unsqueeze(0) # <--- 2 dim mask is (HW), uncomment for (CHW; C = 1) both are valid.

        # return masks and images must have channel dim first (if there is one).
        # return var names must match Collate_Fn.transform's Albumentations.Compose.addional_targets.keys()
        # if use_tensordict and transform is not None.
        # no operations allowed in return statement if use_tensordict.
        return (
            image,
            mask,
            mask_2d,
            binary_label,
            multi_label,
        )


from tensordict_packages.tensordict_wrapper import dataset_to_tensordict
def get_loader(
        batch_size=8,
        image_size=512,
        DEVICE='cpu',
        path_to_dataset_folder=None,
        use_tensordict=False,
        collate_fn=None,
        shuffle=True,
        pin_memory=True,
):

    ds = SomePyTorchDataset(
        batch_size=batch_size,
        image_size=image_size,
        path_to_dataset_folder=path_to_dataset_folder,
        use_tensordict=use_tensordict,
    )
    if use_tensordict:
        ds = dataset_to_tensordict(
            ds=ds,
            DEVICE=DEVICE,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn if use_tensordict else None,
    )
