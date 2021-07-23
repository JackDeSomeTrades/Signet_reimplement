import numpy as np
import pandas as pd
import os
import sys
import torch
from PIL import Image


class SiameseDataset():
    def __init__(self, training_csv=None, train_dir=None, transform=None):
        super(SiameseDataset, self).__init__()
        self.train_set = pd.read_csv(training_csv)
        self.train_set.columns = ["image01", "image02", "label"]
        self.train_dir = train_dir
        self.transform = transform

    def __getitem__(self, index):
        image_path01 = os.path.join(self.train_dir, self.train_set.iat[index, 0])
        image_path02 = os.path.join(self.train_dir, self.train_set.iat[index, 1])

        img0 = Image.open(image_path01)
        img1 = Image.open(image_path02)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.train_set.iat[index, 2])], dtype=np.float32))

    def __len__(self):
        return len(self.train_set)


def testClass():
    train_dir = '/home/pavan/Data/sign_data/train/'
    train_csv = '/home/pavan/Data/sign_data/train_data.csv'
    Dataset = SiameseDataset(training_csv=train_csv, train_dir=train_dir)


if __name__ == '__main__':
    testClass()