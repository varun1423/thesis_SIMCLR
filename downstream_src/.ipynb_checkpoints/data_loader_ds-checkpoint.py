import pandas as pd
import os, random
from IPython.display import clear_output, Image, SVG
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms



class ThermalBarrierCoating(Dataset):
    def __init__(self, phase, train_dir, csv_file, data_dir, crop_size, channel):
        self.train_dir = train_dir
        self.csv_file = csv_file
        self.phase = phase
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.channel = channel
        self.data_csv = pd.read_csv(self.train_dir / Path(self.csv_file))
        self.phase = phase
        self.label = self.data_csv['encoding']
        self.image_ID = self.data_csv['Image_Name']
        self.transforms = tbc_transforms(crop_s=self.crop_size, channel=self.channel, phase=self.phase)

    def __getitem__(self, idx):

        x = Image.open(os.path.join(self.train_dir, self.data_dir,self.image_ID[idx]))
        y = self.label[idx]
        x = self.transforms(x)

        return x, y

    def __len__(self):
        return self.data_csv['encoding'].shape[0]    

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def tbc_transforms(crop_s,
                   channel,
                   horizontal_flip_prob=0.5,
                   phase="train"
                   ):
    if phase == "train":
        if channel == 1:
            transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1),
                 transforms.RandomCrop(crop_s),
                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                 transforms.ToTensor(),
                 ]
            )
        else:
            transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=3),
                 transforms.RandomCrop(crop_s),
                 transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                 transforms.ToTensor(),
                 ]
            )
    else:
        if channel == 1:
            transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1),
                 transforms.RandomCrop(crop_s),
                 transforms.ToTensor(),
                 ]
            )
        else:
            transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=3),
                 transforms.RandomCrop(crop_s),
                 transforms.ToTensor(),
                 ]
            )
    return transform