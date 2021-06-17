import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, io
import sys
import random
import albumentations

def rdir(rootdir):
    for folder, _, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                yield os.path.join(folder, filename)


class CustomTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, dir, keys=None, crop_size=None, coord=False):
        self.transform = transforms.Compose([transforms.Resize(size)])
        self.data = sorted(rdir(dir))
        random.seed(0)
        random.shuffle(self.data)
        self.data = self.data[:-100]
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = {}
        ex["image"] = io.read_image(self.data[i]).squeeze(0)
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = 0
        ex["image"] = ex["image"].unsqueeze(-1).repeat(1, 1, 3)
        return ex

class CustomVal(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, dir, keys=None, crop_size=None, coord=False):
        self.transform = transforms.Compose([transforms.Resize(size)])
        self.data = sorted(rdir(dir))
        random.seed(0)
        random.shuffle(self.data)
        self.data = self.data[-100:]
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = {}
        ex["image"] = io.read_image(self.data[i]).squeeze(0)
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = 0
        ex["image"] = ex["image"].unsqueeze(-1).repeat(1, 1, 3)
        return ex