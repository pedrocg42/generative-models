import os
import numpy as np
from glob import glob

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class FFHQDataset(Dataset):
    def __init__(
        self, split: str, train_val_split: float = 0.9, transform: bool = False, shuffle: bool = True, seed: int = 42
    ):

        self.split = split
        self.train_val_split = train_val_split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed

        # Data loading
        self.image_paths = np.array(glob(os.path.join(config.ffhq_128_image_folder_path, "*", "*.png")))

        # Creating splits
        num_images = len(self.image_paths)

        # Shuffling images
        if self.shuffle:
            np.random.seed(self.seed)
            idx = np.arange(num_images)
            np.random.shuffle(idx)
            self.image_paths = self.image_paths[idx]

        num_images_train = int(num_images * self.train_val_split)
        if self.split == "train":
            self.image_paths = self.image_paths[:num_images_train]
        elif self.split == "val":
            self.image_paths = self.image_paths[num_images_train:]

        if self.transform:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                    transforms.Resize(128),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = torchvision.io.read_image(self.image_paths[index])

        if self.transform:
            image = self.transforms(image)

        # Normalization to [-1., 1.]
        image = image.type(dtype=torch.float32)
        image = (image / 127.5) - 1.0

        return image
