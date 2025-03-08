import json
import os

import imageio
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASS_NAMES = ["01Valve","03Cup","02Tube","04USB","05Joint","08Cube","06PaperCup","07Lighter","09Lamp","10Bolt","11Filter","12Wand","13Wheel","14Bearing"]

class MIPrealDatasetO(Dataset):
    def __init__(self, dataset_path, class_name='15', resize=256):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(
            class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.size = resize

        self.x, self.y, self.mask = self.load_dataset_folder()

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = imageio.imread(x)
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.uint8)


        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = imageio.read(mask)
            mask = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(
            self.dataset_path, self.class_name, 'ground_truth')
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith('.jpg')])
            x.extend(img_fpath_list) # test path

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[
                    0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
    
class MIPrealDataset(Dataset):
    def __init__(self, args, class_name='15', resize=256):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(
            class_name, CLASS_NAMES)
        self.source_path = args.source_path
        self.reflection_path = args.reflection_path
        self.model_path = args.model_path
        self.Rmodel_path = args.Rmodel_path
        self.class_name = class_name
        self.size = resize

        self.x, self.y, self.mask, self.img = self.load_dataset_folder()

    def __getitem__(self, idx):
        x, y, mask, img = self.x[idx], self.y[idx], self.mask[idx], self.img[idx]
        x = imageio.imread(x)
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.uint8)
        img = imageio.imread(img)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.uint8)
        #dep = imageio.imread(dep)
        #dep = cv2.resize(dep, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.uint8)


        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = imageio.read(mask)
            mask = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return x, y, mask, img

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'test'
        x, y, mask, img = [], [], [], []

        Rimg_dir = os.path.join(self.reflection_path, phase)
        gt_dir = os.path.join(self.reflection_path, 'ground_truth')
        img_dir = os.path.join(self.source_path, phase)
        #dep_dir = os.path.join(self.dataset_path, self.class_name, 'depth')
        img_types = sorted(os.listdir(Rimg_dir))
        for img_type in img_types:

            Rimg_type_dir = os.path.join(Rimg_dir, img_type)
            if not os.path.isdir(Rimg_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(Rimg_type_dir, f)
                                    for f in os.listdir(Rimg_type_dir)
                                    if f.endswith('.jpg')])
            x.extend(img_fpath_list) # test path

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith('.jpg')])
            img.extend(img_fpath_list) # test path

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[
                    0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask), list(img)
