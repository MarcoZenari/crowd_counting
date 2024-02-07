import numpy as np

import random
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models

import cv2
from PIL import Image,ImageFilter,ImageDraw
from PIL import ImageStat

class listDataset(Dataset):

    def __init__(self, root_img = str, root_den = str, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            random.seed(1)
            root_img = root_img *4 #data augmentations
            random.shuffle(root_img)

            random.seed(1)
            root_den = root_den *4
            random.shuffle(root_den)


        self.nSamples = len(root_img)
        self.lines_img = root_img
        self.lines_den = root_den
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        train_load = self.train
        img_path = self.lines_img[index]
        den_path = self.lines_den[index]
        img,target = self.load_data(img_path, den_path, train_load)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def load_data(self, img_path, den_path, train_load):
    
        img = Image.open(img_path).convert('RGB')
        den_file = np.load(den_path)
        target = den_file
        if train_load:
            crop_size = (int(img.size[0]/2), int(img.size[1]/2))                  #cropping
            dx = int(random.random()*img.size[0]*1./2)                 #where to start the crop x-axis
            dy = int(random.random()*img.size[1]*1./2)                 #where to start the crop y-axis

            img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
            target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]

            if random.random()>0.8:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        target = cv2.resize(target,
                            (int(target.shape[1]/8),int(target.shape[0]/8)), 
                            interpolation = cv2.INTER_CUBIC)*64

        return img, target
