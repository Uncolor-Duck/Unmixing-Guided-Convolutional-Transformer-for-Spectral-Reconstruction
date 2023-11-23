from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import glob
import os
import torch
import scipy
import scipy.io

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

class MixDatasetVal(Dataset):
    def __init__(self, args, sr='/home/data/duanshiyao/USGS_Library/librarySR.mat', mode='val'):
        if mode != 'val':
            raise Exception("Invalid mode!", mode)
        data_path = args
        self.mat = data_path
        data_names = os.listdir(self.mat)
        self.keys = data_names
        random.shuffle(data_names)
        self.keys = [self.mat + data_names[i] for i in range(len(self.keys))]

        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        #mat = h5py.File(self.keys[index], 'r')
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['hyper']))
        #hyper = normalize(hyper, max_val=1., min_val=0.)
        hyper = np.transpose(hyper, [2, 0, 1])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

class MixDataSetTrain(Dataset):
    def __init__(self, args, sr="/home/data/duanshiyao/USGS_Library/librarySR.mat", mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = args
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)
    def __len__(self):
        return len(self.keys)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img


    def __getitem__(self, index):
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 0, 1])

        hyper = np.ascontiguousarray(self.arguement(hyper, rotTimes, vFlip, hFlip))
        hyper = torch.Tensor(hyper)


        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = np.ascontiguousarray(self.arguement(rgb, rotTimes, vFlip, hFlip))
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

class NewDatasetVal(Dataset):
    def __init__(self, args, sr='/home/data/duanshiyao/USGS_Library/librarySR.mat', mode='val'):
        if mode != 'val':
            raise Exception("Invalid mode!", mode)
        data_path = args
        self.mat = data_path
        data_names = os.listdir(self.mat)
        self.keys = data_names
        random.shuffle(data_names)
        self.keys = [self.mat + data_names[i] for i in range(len(self.keys))]

        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        #mat = h5py.File(self.keys[index], 'r')
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        #hyper = normalize(hyper, max_val=1., min_val=0.)
        hyper = np.transpose(hyper, [2, 0, 1])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

class NewDataSetTrain(Dataset):
    def __init__(self, args, sr="/home/data/duanshiyao/USGS_Library/librarySR.mat", mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = args
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)
    def __len__(self):
        return len(self.keys)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img


    def __getitem__(self, index):
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 0, 1])

        hyper = np.ascontiguousarray(self.arguement(hyper, rotTimes, vFlip, hFlip))
        hyper = torch.Tensor(hyper)


        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = np.ascontiguousarray(self.arguement(rgb, rotTimes, vFlip, hFlip))
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

def main():
    dataset = NewDatasetVal(args="/home/data/duanshiyao/Unmixing/PDASS/New/Val/")
    X, Y = dataset[0]
    hh = 1


if __name__ == "__main__":
    main()


