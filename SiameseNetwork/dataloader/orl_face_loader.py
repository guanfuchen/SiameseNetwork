#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.ImageOps
from PIL import Image
import random
import numpy as np

class OrlFaceLoader(Dataset):
    def __init__(self, root, transform=None, should_invert=True):
        self.imageFolderDataset = ImageFolder(root=root)
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
                                     transforms.Scale((100,100)),
                                     transforms.ToTensor()])
    batch_size = 4
    dst =  OrlFaceLoader(root = os.path.expanduser('~/Data/orl_faces/train'),
                                 transform = transform,
                                 should_invert = False)
    trainloader = DataLoader(dst, batch_size=batch_size)
    for i, (img0s, img1s, labels) in enumerate(trainloader):
        print(i)
        # print(img0s.shape)
        # print(img1s.shape)
        # print(labels.shape)
        # 标签为0代表相同，为1代表不同
        img0s_np = img0s.numpy()
        img1s_np = img1s.numpy()
        labels_np = labels.numpy()
        img0_0 = img0s_np[0][0]
        img1_0 = img1s_np[0][0]
        label_0 = labels_np[0]

        ax1 = plt.subplot(121)
        ax1.imshow(img0_0, cmap='gray')
        bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
        ax1.text(0, 0, label_0, ha="center", va="center", rotation=45,
                size=15,
                bbox=bbox_props)
        ax2 = plt.subplot(122)
        ax2.imshow(img1_0, cmap='gray')
        plt.show()
