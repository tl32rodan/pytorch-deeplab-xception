from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import pandas as pd


class AICUPDataset(Dataset):
    """
    AICup 2021 rice plant objection detection dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('aicup'),
                 split='Train_Dev',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir

        self.args = args

        self.task = args.task

        self.split = split

        self.im_ids = []
        self.images = []
        self.labels = []
        self.sigma = 10 # determine the sigma of confidence map
        self.range = 2 # determine the range of label points
        self.dis = 20 # distance constraint

        if split == 'Train_Dev':
            for img in os.listdir(os.path.join(base_dir, split, 'training')):
                im_id = img[:-4]
                
                label = os.path.join(base_dir, split, 'train_labels', im_id +'.csv')
                self.im_ids.append(im_id)
                self.images.append(os.path.join(base_dir, split, 'training', img))
                self.labels.append(label)

            assert (len(self.images) == len(self.labels))
        else:
            for img in os.listdir(os.path.join(base_dir, split)):
                im_id = img[:-4]
                
                self.im_ids.append(im_id)
                self.images.append(os.path.join(base_dir, split, img))
            

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _label = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _label}

        if self.split == 'Train_Dev':
            sample = self.transform_tr(sample)
            sample['label'] /= 255
            return sample
        else:
            sample = self.transform_val(sample)
            sample['label'] /= 255
            return sample


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = np.zeros(_img.size)
        if self.split != 'Train_Dev':
            return _img,  Image.fromarray(np.uint8(_label))

        _points = pd.read_csv(self.labels[index], header=None).to_numpy()
        if self.task == 'segmentation':
            for (x, y) in _points:
                _label[x, y] = 1.
                #for i in range(-self.range, self.range):
                #    for j in range(-self.range, self.range):
                #        if x+i >= _label.shape[0] or y+j >= _label.shape[1] or x+i<0 or y+j<0 or (i**2+j**2)**0.5 > self.dis:
                #            continue
                #        _label[x+i, y+j] = 1
        elif self.task == 'regression':
            for (x, y) in _points:
                for i in range(-self.range, self.range):
                    for j in range(-self.range, self.range):
                        if x+i >= _label.shape[0] or y+j >= _label.shape[1] or x+i<0 or y+j<0 :
                            continue
                        gaussian_val = (np.exp(-0.5*(i**2+j**2)/(self.sigma)))
                        _label[x+i, y+j] = gaussian_val
        else:
            raise ValueError

        
        _label = Image.fromarray(np.uint8(np.transpose(_label)*255))
        return _img, _label

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'AICUP2021(split=' + str(self.split) + ')'
