import pandas as pd
import os, random
from IPython.display import clear_output, Image, SVG
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def tbc_transforms(horizontal_flip_prob=0.5,
                   gaussian_prob=0.5,
                   crop_s=224,
                   ):
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
         transforms.RandomResizedCrop(crop_s),
         transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
         transforms.RandomRotation(30),
         transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
         transforms.ToTensor(),
         ]
    )
    return transform


class ThermalBarrierCoating(Dataset):
    def __init__(self,phase, train_dir, csv_file, data_dir, crop_size):
        self.train_dir = train_dir
        self.csv_file = csv_file
        self.phase = phase
        self.data_dir = data_dir
        self.crop_size = crop_size
        if self.phase == 'train':
            self.data_csv = pd.read_csv(self.train_dir / Path(self.csv_file))
        else:
            self.data_csv = pd.read_csv(self.train_dir / Path(self.csv_file))

        self.phase = phase
        self.label = self.data_csv['encoding']
        self.image_ID = self.data_csv['Image_Name']
        self.transforms = tbc_transforms(crop_s = crop_size)

    def __getitem__(self, idx):

        #x = self.imgarr[idx]
        x = Image.open(os.path.join(self.train_dir, self.data_dir,self.image_ID[idx]))
        # print(x.shape)
        #x = x.astype(np.float32) / 255.0

        x1 = self.augment(x)
        x2 = self.augment(x)

        return x1, x2

    def __len__(self):
        return self.data_csv['encoding'].shape[0]


    def augment(self, img):

        if self.phase == 'train':
            img = self.transforms(img)
        else:
            return img

        return img
