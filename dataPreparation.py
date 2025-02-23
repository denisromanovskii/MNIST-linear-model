import torch
import torchvision
from torchvision.transforms import v2

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class MNISTDataset(Dataset):
    """ Класс для подготовки датасета MNIST """
    def __init__(self, path, transform=None):
        self.path = path
        self.transorm = transform

        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_idx = { # one-hot-number
                    cls_name : i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = path_dir.split('/')[-1].split('\\')[-1]

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))
            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = Image.open(file_path)
        if self.transorm is not None:
            sample = self.transorm(sample)

        return sample, target

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, ), std=(0.5, ))
])

train_data = MNISTDataset('mnist/training', transform=transform)
test_data = MNISTDataset('mnist/testing', transform=transform)

train_data, val_data = random_split(train_data, [0.7, 0.3])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True)