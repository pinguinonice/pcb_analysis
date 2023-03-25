# train_unet.py

from patchify import patchify, unpatchify
import random
from keras.metrics import MeanIoU
from sklearn.utils import class_weight
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from unet_model import multi_unet_model  # Uses softmaxcd

from keras.callbacks import TensorBoard
import datetime

from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pcb_analysis import PCB, DATASET

import datetime

from image_generator import load_img, make_patches, get_random_image_and_mask_patches, imageLoader
batch_size = 128


def get_model(size_x, size_y):
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=size_x, IMG_WIDTH=size_y, IMG_CHANNELS=3)


train_img_dir = "dataset/pcb_dataset/image"
train_mask_dir = "dataset/pcb_dataset/icmask"
train_img_list = os.listdir(train_img_dir)  # get the image list
train_mask_list = os.listdir(train_mask_dir)


# remove all non jpg/png files and .* files
train_img_list = [x for x in train_img_list if 'jpg' in x.split(
    '.')[-1] or 'png' in x.split('.')[-1]]
train_mask_list = [x for x in train_mask_list if 'jpg' in x.split(
    '.')[-1] or 'png' in x.split('.')[-1]]

# sort the two list in the same order
train_img_list.sort()
train_mask_list.sort()

batch_size = 128
epochs = 1

size_x, size_y = 256, 256  # size of path and unet
r, c = 10, 6  # rows and cols for patches
image_size = (r*size_x, c*size_y)  # size for resize so patches fit

n_input_band = 3
classes = {'bg': [0, 0, 0], 'ic': [0, 1, 0], 'pcb': [1, 1, 1]}
n_classes = len(classes)


train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size, image_size, n_input_band, n_classes, size_x, size_y, classes)

# vali_img_datagen = imageLoader(vali_img_dir, vali_img_list,
#                                vali_mask_dir, vali_mask_list, batch_size, image_size, n_input_band, n_classes, size_x, size_y)

model = get_model(size_x, size_y)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# # load weights
path_weights = 'unet-ic-pcb_10x256x6x256_2023-03-23 23:52:40.877297.hdf5'
while True:
    model.load_weights(path_weights)

    # fit the model
    model.fit(train_img_datagen,
              steps_per_epoch=(len(train_img_list)*r*c)//batch_size,
              epochs=epochs)

    path_weights = 'unet-ic-pcb_{}x{}x{}x{}_{}.hdf5'.format(
        r, size_x, c, size_y, datetime.datetime.now())

    model.save(path_weights)
