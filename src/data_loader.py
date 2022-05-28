import random
from pathlib import Path
from . import functions_file
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import wandb


class iter_data_loader():
    def __init__(self,
                 npy_file,
                 batch_size,
                 crop_size,
                 channel,
                 train_dir,
                 cropping_strategy,
                 rotation_prob,
                 blur_prob,
                 horizontal_flip_prob):

        self.npy_file = npy_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.channel = channel
        self.train_dir = train_dir
        self.cropping_strategy = cropping_strategy
        self.rotation_prob = rotation_prob
        self.blur_prob = blur_prob
        self.horizontal_flip_prob = horizontal_flip_prob

        self.all_train_data_images_class_wise_list = np.load(self.train_dir/Path(self.npy_file), allow_pickle=True) #load data

        # initialize transfrom
        self.transforms = functions_file.tbc_transform(channel=self.channel,
                                                    rotation_prob=self.rotation_prob,
                                                    blur_prob=self.blur_prob,
                                                    horizontal_flip_prob=self.horizontal_flip_prob)

        self.transform_crop = transforms.Compose([transforms.RandomCrop(self.crop_size)])
        self.anchor = transforms.Compose([transforms.RandomCrop(256)]) #needed for baseline 
        wandb.log({"Transform_pipeline": self.transforms, "crop_transform": self.transform_crop})

    def __iter__(self):
        z_1 = torch.tensor([])
        z_2 = torch.tensor([])
        if self.cropping_strategy == "crops_overlap_and_non_overlap":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]
                overlap = random.choice([True, False])

                x_1, x_2 = functions_file.positive_pair(img_orig,
                                                        self.crop_size,
                                                        overlap,
                                                        img_orig.shape[1],
                                                        img_orig.shape[0])

                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)

        elif self.cropping_strategy == "crops_no_overlap_only":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]
                overlap = False

                x_1, x_2 = functions_file.positive_pair(img_orig,
                                                        self.crop_size,
                                                        overlap,
                                                        img_orig.shape[1],
                                                        img_orig.shape[0])

                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)

        elif self.cropping_strategy == "crops_overlap_only":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]
                overlap = True

                x_1, x_2 = functions_file.positive_pair(img_orig,
                                                        self.crop_size,
                                                        overlap,
                                                        img_orig.shape[1],
                                                        img_orig.shape[0])

                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)

        elif self.cropping_strategy == "crops_overlap_50_percent":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]

                x_1, x_2 = functions_file.positive_pair_50_overlap(img_orig,
                                                                   self.crop_size,
                                                                   overlap,
                                                                   img_orig.shape[1],
                                                                   img_orig.shape[0])

                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)

        elif self.cropping_strategy == "multi_instance":
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

        elif self.cropping_strategy == "baseline":
            for i in range(self.batch_size):
                random_class_idx = random.randint(0, 199)
                img_orig = self.all_train_data_images_class_wise_list[random_class_idx]
                img_orig = Image.fromarray(img_orig)
                x = self.anchor(img_orig)
                x_1 = self.transform_crop(x)
                x_2 = self.transform_crop(x)
                x_1 = self.transforms(x_1)
                x_2 = self.transforms(x_2)
                x_1 = torch.reshape(x_1, (-1, x_1.size()[0], x_1.size()[1], x_1.size()[2]))
                x_2 = torch.reshape(x_2, (-1, x_2.size()[0], x_2.size()[1], x_2.size()[2]))
                z_1 = torch.cat((z_1, x_1), 0)
                z_2 = torch.cat((z_2, x_2), 0)
            yield (z_1, z_2)
        else:
            raise NotImplementedError


