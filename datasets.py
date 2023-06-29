import os
import cv2
import random
import numpy as np

from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class myCaltech101(Dataset):
    def __init__(self, data_folder="./Caltech101/caltech101/101_ObjectCategories",
                       phase="train",
                       crop_dim=224,
                       transform=None):
        """
          input:
            - data_folder: path of Caltech101 dataset
            - phase: train or val
            - img_size:
                + train: resize to (256, _) (crop to 224) and resize to (192, _) (crop to 180)
                + val: resize to (256, _) (crop to 224)
        """
        self.data_folder = data_folder
        self.phase = phase
        self.crop_dim = crop_dim
        self.transform = transform

        self.list_class_index = []
        self.list_image_path = []
        list_sub_folder = os.listdir(self.data_folder)
        list_sub_folder.sort()
        list_sub_folder = list_sub_folder[1:]
        for i, sub_folder in enumerate(list_sub_folder):
            sub_folder_path = os.path.join(data_folder, sub_folder)
            list_file_name = os.listdir(sub_folder_path)
            list_file_name.sort()
            for file_name in list_file_name:
                if file_name.endswith(".jpg"):
                    image_path = os.path.join(sub_folder_path, file_name)
                    self.list_image_path.append(image_path)
                    self.list_class_index.append(i)

    def __len__(self):
        return len(self.list_image_path)

    def __getitem__(self, index):
        label = self.list_class_index[index]
        image_path = self.list_image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        min_resized = 0
        if self.phase == 'val':
            self.crop_dim = 224
            min_resized = 256
        else:
            if self.crop_dim == 224:
                min_resized = 256
            elif self.crop_dim == 180:
                min_resized = 192

        img_height, img_width, img_channel = image.shape

        aspect_ratio = img_height / img_width # Keep the aspect ratio same

        if img_height < img_width:
            img_height = min_resized
            img_width = int(img_height / aspect_ratio)
        else:
            img_width = min_resized
            img_height = int(img_width * aspect_ratio)
        image = cv2.resize(image,(img_width, img_height))

        h, w, c = image.shape
        image = image[int(h/2-self.crop_dim/2):int(h/2+self.crop_dim/2) , int(w/2-self.crop_dim/2):int(w/2+self.crop_dim/2)] # Crop the middle view

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        return image, label
    

class Caltech101Loader:
    def __init__(self, data_folder="./Caltech101/caltech101/101_ObjectCategories",
                       phase="train",
                       crop_dim=224,
                       batch_size=128,
                       shuffle=True,
                       random_seed=42,
                       valid_size=0.2):
        self.data_folder = data_folder
        self.phase = phase
        self.crop_dim = crop_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.valid_size = valid_size

    def load_data(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        valid_transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = myCaltech101(data_folder=self.data_folder, phase="train", crop_dim=self.crop_dim, transform=train_transform)
        valid_dataset = myCaltech101(data_folder=self.data_folder, phase="val", crop_dim=self.crop_dim, transform=valid_transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.seed(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, sampler=train_sampler
        )

        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=self.batch_size, sampler=valid_sampler
        )

        return (train_loader, valid_loader)