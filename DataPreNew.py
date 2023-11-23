import os
import os.path
import h5py
# from scipy.io import loadmat
import cv2
import glob
import numpy as np
import argparse
import hdf5storage
import random
import scipy.io
import h5py


parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--data_path", type=str, default='/home/data/duanshiyao/Unmixing/PDASS_Data/Train/', help="data path")
parser.add_argument("--val_path", type=str, default='/home/data/duanshiyao/Unmixing/PDASS_Data/Val/', help="data path")
parser.add_argument("--patch_size", type=int, default=128, help="data patch size")
parser.add_argument("--stride", type=int, default=64, help="data patch stride")
parser.add_argument("--train_data_path", type=str, default='/home/data/duanshiyao/DCD_dataset/grss/train/rgb/', help="preprocess_data_path")
parser.add_argument("--val_data_path", type=str, default='/home/data/duanshiyao/DCD_dataset/grss/val/', help="preprocess_data_path")
opt = parser.parse_args()


def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def process_data(patch_size, stride, mode="train"):
    if mode == 'train':
        print("\nprocess training set ...\n")
        patch_num = 1
        filenames_hyper = glob.glob(os.path.join(opt.data_path, '*.mat'))
        filenames_val = glob.glob(os.path.join(opt.val_path, '*.mat'))
        filenames_hyper.sort()
        filenames_val.sort()
        # for k in range(1):  # make small dataset
        for k in range(len(filenames_hyper)//2, len(filenames_hyper)):
            print(filenames_hyper[k])
            # load hyperspectral image
            try:
                mat = scipy.io.loadmat(filenames_hyper[k])
            except:
                mat = h5py.File(filenames_hyper[k], 'r')

            #mat = scipy.io.loadmat(filenames_hyper[k])

            hyper = np.float32(np.array(mat['hyper']))
            #hyper = normalize(hyper, max_val=1., min_val=0.)  $Normalization Done In Preprocessing$
            hyper = np.transpose(hyper, [2, 0, 1])

            rgb = np.float32(np.array(mat['rgb']))
            rgb = np.transpose(rgb, [2, 0, 1])
            C, W, H = hyper.shape
            hyper_train = hyper
            RGB_train = rgb
            #hyper_val = hyper[:, :, -H//4:]
            #RGB_val = rgb[:, :, -H//4:]
            patches_hyper_train = Im2Patch(hyper_train, win=patch_size, stride=stride)
            patches_rgb_train = Im2Patch(RGB_train, win=patch_size, stride=stride)
            # add data ：重组patches
            for j in range(0, patches_hyper_train.shape[3]):
                print("generate training sample #%d" % patch_num)
                sub_hyper = patches_hyper_train[:, :, :, j]
                sub_rgb = patches_rgb_train[:, :, :, j]

                train_data_path_array = opt.train_data_path
                train_data_path = os.path.join(train_data_path_array, 'train' + str(patch_num) + '.mat')
                hdf5storage.savemat(train_data_path, {'cube': sub_hyper}, format='7.3')
                hdf5storage.savemat(train_data_path, {'rgb': sub_rgb}, format='7.3')

                patch_num += 1


        for k in range(len(filenames_val)):
            print(filenames_val[k])
            # load hyperspectral image
            mat = h5py.File(filenames_hyper[k], 'r')

            #mat = scipy.io.loadmat(filenames_val[k])

            hyper = np.float32(np.array(mat['hyper']))
            #hyper = normalize(hyper, max_val=1., min_val=0.)  $Normalization Done In Preprocessing$
            hyper = np.transpose(hyper, [2, 0, 1])

            rgb = np.float32(np.array(mat['rgb']))
            rgb = np.transpose(rgb, [2, 0, 1])
            C, W, H = hyper.shape
            hyper_train = hyper
            RGB_train = rgb
            #hyper_val = hyper[:, :, -H//4:]
            #RGB_val = rgb[:, :, -H//4:]
            train_data_path_array = opt.val_data_path
            train_data_path = os.path.join(train_data_path_array, 'val' + str(k) + '.mat')
            hdf5storage.savemat(train_data_path, {'cube': hyper}, format='7.3')
            hdf5storage.savemat(train_data_path, {'rgb': rgb}, format='7.3')

        print("\ntraining set: # samples %d\n" % (patch_num - 1))


def main():
    if not os.path.exists(opt.train_data_path):
        os.makedirs(opt.train_data_path)
    process_data(patch_size=opt.patch_size, stride=opt.stride, mode='train')

if __name__ == "__main__":
    main()
