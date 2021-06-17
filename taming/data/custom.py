import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, io
import sys
import random

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        img = io.read_image(ex)
        ex["image"] = self.transform(img)
        ex["class"] = 0
        return ex

class CustomTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, dir, keys=None, crop_size=None, coord=False):
        self.transform = transforms.Compose([transforms.Resize(size)])
        self.data = sorted(rdir(dir))
        random.seed(0)
        random.shuffle(self.data)
        self.data = self.data[-100:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        img = io.read_image(ex)
        ex["image"] = self.transform(img)
        ex["class"] = 0
        return ex