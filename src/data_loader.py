import pandas as pd
import os, random
from IPython.display import clear_output, Image, SVG
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from . import functions_file
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py as h5
import numpy as np

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


def tbc_transforms(crop_s,
                   channel,
                   horizontal_flip_prob=0.5,
                   gaussian_prob=0.5,
                   rotation_prob=0.5,
                   solarization_prob=0.5,
                   color_jitter_prob=0.5,
                   brightness=0.8,
                   contrast=0.8,
                   saturation=0.8,
                   hue=0.2,
                   min_scale=0.08,
                   max_scale=1.0,
                   ):
    if channel == 1:
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
             transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
             transforms.RandomApply([transforms.RandomRotation(30)], p=rotation_prob),
             transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
             ]
        )
    elif channel == 3:
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.RandomCrop(crop_s),
             transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
             transforms.RandomApply([transforms.RandomRotation(30)], p=rotation_prob),
             transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
             ]
        )
    else:
        transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=3),
                 transforms.RandomResizedCrop(
                    crop_s,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                    transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                    ),
                    transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                    transforms.RandomApply([Solarization()], p=solarization_prob),
                    transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
                ]
            )
    return transform


def tbc_transforms_npy_loader(channel,
                              horizontal_flip_prob=0.5,
                              gaussian_prob=0.5,
                              rotation_prob=0.5,
                              solarization_prob=0.5,
                              color_jitter_prob=0.5,
                              brightness=0.8,
                              contrast=0.8,
                              saturation=0.8,
                              hue=0.2,
                              min_scale=0.08,
                              max_scale=1.0,
                              ):
    if channel == 1:
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
             transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
             transforms.RandomApply([transforms.RandomRotation(30)], p=rotation_prob),
             transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
             ]
        )
    elif channel == 3:
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
             transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
             transforms.ToTensor(),
             ]
        )
    else:
        print("------NOT_IMPLEMENTED------")
    return transform

"""
transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
             transforms.RandomApply([transforms.RandomRotation(30)], p=rotation_prob),
             transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
             ]
        )
"""

class ThermalBarrierCoating(Dataset):
    def __init__(self, phase, train_dir, csv_file, data_dir, crop_size, channel, cropping_strategy):
        self.train_dir = train_dir
        self.csv_file = csv_file
        self.phase = phase
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.channel = channel
        self.cropping_strategy = cropping_strategy
        #if self.phase == 'train':
        self.data_csv = pd.read_csv(self.train_dir / Path(self.csv_file))
        #else:
        #    self.data_csv = pd.read_csv(self.train_dir / Path(self.csv_file))

        self.phase = phase
        self.label = self.data_csv['encoding']
        self.image_ID = self.data_csv['Image_Name']
        self.image_name = self.data_csv['original_img_name']
        self.transforms = tbc_transforms(crop_s=self.crop_size, channel=self.channel)

    def __getitem__(self, idx):
        #x = self.imgarr[idx]
        if self.cropping_strategy == "baseline":
            x = Image.open(os.path.join(self.train_dir,
                                        self.data_dir,
                                        self.image_ID[idx]))
            x1 = self.augment(x)
            x2 = self.augment(x)
        elif self.cropping_strategy == "crops_overlap_and_non_overlap":
            x = Image.open(os.path.join(self.train_dir,
                                        self.data_dir,
                                        self.image_name[idx]), mode='r')
            x_shape = np.array(x)
            if x_shape.shape[0] == 2048:
                overlap = random.choice([True, False])
                x_1, x_2 = functions_file.random_crop_2048(os.path.join(self.train_dir, self.data_dir, self.image_name[idx]), self.crop_size, overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)
            else:
                overlap = random.choice([True, False])
                x_1, x_2 = functions_file.random_crop_712(os.path.join(self.train_dir, self.data_dir, self.image_name[idx]), self.crop_size, overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)
        else:
            print("________________NOT_IMPLEMENTED_YET___________")

        return x_1, x_2

    def __len__(self):
        return self.data_csv['original_img_name'].shape[0]


    def augment(self, img):

        if self.phase == 'train':
            img = self.transforms(img)
        else:
            return img

        return img


class NPY_loader(Dataset):
    def __init__(self, train_dir, npy_file, crop_size, channel, cropping_strategy):
        self.train_dir = train_dir
        self.npy_file = npy_file
        #self.data_dir = data_dir
        self.crop_size = crop_size
        self.channel = channel
        self.cropping_strategy = cropping_strategy

        self.train_list = np.load(self.train_dir/Path(self.npy_file), allow_pickle=True)
        self.transforms = tbc_transforms_npy_loader(crop_s=self.crop_size,
                                                    channel=self.channel)

    def __getitem__(self, idx):

        if self.cropping_strategy == "crops_overlap_and_non_overlap":
            if self.train_list[idx].shape[0] == 2048:
                overlap = random.choice([True, False])
                x_1, x_2 = functions_file.random_crop_2048(self.train_list[idx],
                                                           self.crop_size,
                                                           overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)
            else:
                overlap = random.choice([True, False])
                x_1, x_2 = functions_file.random_crop_712(self.train_list[idx],
                                                           self.crop_size,
                                                           overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)

        elif self.cropping_strategy == "crops_no_overlap_only":
            if self.train_list[idx].shape[0] == 2048:
                overlap = False
                x_1, x_2 = functions_file.random_crop_2048(self.train_list[idx],
                                                           self.crop_size,
                                                           overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)
            else:
                overlap = False
                x_1, x_2 = functions_file.random_crop_712(self.train_list[idx],
                                                           self.crop_size,
                                                           overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)
        elif self.cropping_strategy == "crops_overlap_only":
            if self.train_list[idx].shape[0] == 2048:
                overlap = True
                x_1, x_2 = functions_file.random_crop_2048(self.train_list[idx],
                                                           self.crop_size,
                                                           overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)
            else:
                overlap = True
                x_1, x_2 = functions_file.random_crop_712(self.train_list[idx],
                                                           self.crop_size,
                                                           overlap)
                x_1 = self.augment(x_1)
                x_2 = self.augment(x_2)
        else:
            print("________________NOT_IMPLEMENTED_YET___________")

        return x_1, x_2

    def __len__(self):
        return len(self.train_list)


    def augment(self, img):

        img = self.transforms(img)

        return img

class iter_data_loader():
    def __init__(self,npy_file, batch_size, crop_size, channel, train_dir,cropping_strategy):
        self.npy_file = npy_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.channel = channel
        self.train_dir = train_dir
        self.cropping_strategy = cropping_strategy
        self.all_train_data_images_class_wise_list = np.load(self.train_dir/Path(self.npy_file), allow_pickle=True)

        self.transforms = tbc_transforms_npy_loader(channel=self.channel)
        self.transform_crop = transforms.Compose([transforms.RandomCrop(self.crop_size)])

    def __iter__(self):
        z_1 = torch.tensor([])
        z_2 = torch.tensor([])
        if self.cropping_strategy == "crops_overlap_and_non_overlap":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]
                overlap = random.choice([True, False])

                if img_orig.shape[0] == 2048:
                    x_1, x_2 = functions_file.random_crop_2048(img_orig,
                                                               self.crop_size,
                                                               overlap)
                else:
                    x_1, x_2 = functions_file.random_crop_712(img_orig,
                                                              self.crop_size,
                                                              overlap)
                x_1 = self.transform_crop(x_1)
                x_2 = self.transform_crop(x_2)
                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)

        if self.cropping_strategy == "crops_no_overlap_only":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]
                overlap = False

                if img_orig.shape[0] == 2048:
                    x_1, x_2 = functions_file.random_crop_2048(img_orig,
                                                               self.crop_size,
                                                               overlap)
                else:
                    x_1, x_2 = functions_file.random_crop_712(img_orig,
                                                              self.crop_size,
                                                              overlap)
                x_1 = self.transform_crop(x_1)
                x_2 = self.transform_crop(x_2)
                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)

        if self.cropping_strategy == "crops_overlap_only":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]
                overlap = True

                if img_orig.shape[0] == 2048:
                    x_1, x_2 = functions_file.random_crop_2048(img_orig,
                                                               self.crop_size,
                                                               overlap)
                else:
                    x_1, x_2 = functions_file.random_crop_712(img_orig,
                                                              self.crop_size,
                                                              overlap)
                x_1 = self.transform_crop(x_1)
                x_2 = self.transform_crop(x_2)
                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)


class multi_instance_data_loader():
    def __init__(self,npy_file, batch_size, crop_size, channel, train_dir):
        self.npy_file = npy_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.channel = channel
        self.train_dir = train_dir
        self.all_train_data_images_class_wise_list = np.load(self.train_dir/Path(self.npy_file), allow_pickle=True)

        self.transforms = tbc_transforms_npy_loader(channel=self.channel)
        self.transform_crop = transforms.Compose([transforms.RandomCrop(self.crop_size)])



    def __iter__(self):
        z_1 = torch.tensor([])
        z_2 = torch.tensor([])

        for i in range(self.batch_size):
            random_class_idx = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            img_class = self.all_train_data_images_class_wise_list[random_class_idx]
            len_img_class = len(img_class)
            img_idx_in_class = list(range(len_img_class))
            img_idx_random_first = random.choice(img_idx_in_class)
            del img_idx_in_class[img_idx_random_first]
            img_idx_random_second = random.choice(img_idx_in_class)

            x_1 = Image.fromarray(self.all_train_data_images_class_wise_list[random_class_idx][img_idx_random_first])
            x_2 = Image.fromarray(self.all_train_data_images_class_wise_list[random_class_idx][img_idx_random_second])
            x_1 = self.transform_crop(x_1)
            x_2 = self.transform_crop(x_2)
            x_1 = self.transforms(x_1)
            x_2 = self.transforms(x_2)
            x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
            x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
            z_1 = torch.cat((z_1, x_1), 0)
            z_2 = torch.cat((z_2, x_2), 0)
        yield (z_1, z_2)


class complementary_masking():
    def __init__(self, npy_file, batch_size, crop_size, channel, train_dir):
        self.train_dir = train_dir
        self.npy_file = npy_file
        self.crop_size = crop_size
        self.channel = channel
        self.batch_size = batch_size

        self.train_list = np.load(self.train_dir/Path(self.npy_file), allow_pickle=True)
        self.transforms = tbc_transforms_npy_loader(channel=self.channel)
        self.transform_crop = transforms.Compose([transforms.RandomCrop(self.crop_size)])

    def __iter__(self):
        z_1 = torch.tensor([])
        z_2 = torch.tensor([])

        mask = np.zeros((self.crop_size,self.crop_size))
        mask[56: 56+(2*56), 56:56+(2*56)]=255

        for i in range(self.batch_size):
            idx = random.randint(0, 200)
            image = self.train_list[idx]
            image = Image.fromarray(image)
            image = self.transform_crop(image)
            image = np.array(image)

            masked_img = np.where(mask == 255, image, 0)
            masked_img_invert = np.where(mask == 0,image,0)
            a = Image.fromarray(masked_img)
            b = Image.fromarray(masked_img_invert)

            x_1 = self.transforms(a)
            x_2 = self.transforms(b)
            x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
            x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
            z_1 = torch.cat((z_1, x_1), 0)
            z_2 = torch.cat((z_2, x_2), 0)
        yield (z_1, z_2)


class TBC_H5(Dataset):
    def __init__(self, hdf5_file, crop_size, channel, device, phase="train"):
        self.phase = phase
        self.hdf5_file = hdf5_file
        self.crop_size = crop_size
        self.channel = channel
        self.transforms = tbc_transforms(crop_s=self.crop_size,
                                         channel=self.channel)
        self.device = device
        with h5.File(hdf5_file, "r") as f:
            self.length = len(f['labels'])
            self.data = f['data'][:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = Image.fromarray(self.data[idx], mode="L")
        x1 = self.augment(x)
        x2 = self.augment(x)
        return x1.to(self.device), x2.to(self.device)


    def augment(self, img):

        if self.phase == 'train':
            img = self.transforms(img)
        else:
            return img
        return img