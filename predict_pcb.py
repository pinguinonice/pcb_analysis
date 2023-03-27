# predict_pcb.py 

from os import listdir
from os.path import isfile, join
import patchify
import random
from keras.metrics import MeanIoU
from sklearn.utils import class_weight
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from unet_model import multi_unet_model  # Uses softmax

from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

from pcb_analysis import PCB, DATASET

path_weights = 'unet-ic-pcb_10x256x6x256_2023-03-27 17:26:09.832389.hdf5'
# getting input size and rows cols for patches from name

parameter = path_weights.split('_')[1].split('x')

SIZE_X = int(parameter[1])  # i.e. 256
SIZE_Y = int(parameter[3])  # i.e. 256
r = int(parameter[0])  # i.e. 5
c = int(parameter[2])  # i.e. 8

n_input_band = 3
n_classes = 3

model = multi_unet_model(
    n_classes=n_classes, IMG_HEIGHT=SIZE_X, IMG_WIDTH=SIZE_Y, IMG_CHANNELS=n_input_band)

# load weights
model.load_weights(path_weights)

test_img_dir = "dataset/pcb_dataset/image_test"
test_mask_dir = "dataset/pcb_dataset/icmask_test"
test_img_paths = os.listdir(test_img_dir)  # get the image list
test_mask_paths = os.listdir(test_mask_dir)



# remove all non jpg/png files and .* files
test_img_paths = [x for x in test_img_paths if 'jpg' in x.split(
    '.')[-1] or 'png' in x.split('.')[-1]]
test_mask_paths = [x for x in test_mask_paths if 'jpg' in x.split(
    '.')[-1] or 'png' in x.split('.')[-1]]

# sort the two list in the same order
test_img_paths.sort()
test_mask_paths.sort()


for image_path, mask_path in zip(test_img_paths, test_mask_paths):

    large_image = cv2.imread(join(test_img_dir, image_path))
    large_mask = cv2.imread(join(test_mask_dir, mask_path))

    shape_original = large_image.shape
    large_image = cv2.resize(large_image, (SIZE_X*r, SIZE_Y*c))

    # patch the image  (r,c,1,SIZE_X, SIZE_Y,3)
    patches = patchify.patchify(large_image, (SIZE_X, SIZE_Y, 3), step=256)

    # reshape for model input (r*c,SIZE_X, SIZE_Y, 3)
    patches_reshape = patches.reshape(
        (patches.shape[0]*patches.shape[1], patches.shape[3], patches.shape[4], patches.shape[5]))

    # normalize
    patches_reshape = patches_reshape.astype(np.float64)/255.

    # predict all at once
    predicted_patches = model.predict(patches_reshape)

    # reshape to patchify convention (r,c,1,SIZE_X, SIZE_Y, 2)
    predicted_patches_reshaped = predicted_patches.reshape(
        (patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4], n_classes))

    reconstructed_predicted = patchify.unpatchify(
        predicted_patches_reshaped, large_image.shape)

    # scale to original size
    reconstructed_predicted_large = cv2.resize(
        reconstructed_predicted.astype(np.float32), shape_original[1::-1])
    large_image = cv2.resize(large_image, shape_original[1::-1])

    # reconstructed_classes_large = np.argmax(
    #     reconstructed_predicted_large, axis=-1)

    # rgb_reconstructed_classes_large = np.stack(
    #     [reconstructed_classes_large*255]*3, axis=-1)
    reconstructed_predicted_large = np.where(
        reconstructed_predicted_large > 0.5, 1, 0)
    

    rgb_reconstructed_classes_large = reconstructed_predicted_large*255

    # # plot
    # plt.figure(figsize=(12, 4))

    # plt.subplot(131)
    # plt.title('Input')
    # plt.imshow(large_image)

    # plt.subplot(132)
    # plt.title('Prediction')
    # plt.imshow(rgb_reconstructed_classes_large)

    # plt.subplot(133)
    # plt.title('Ground Truth')
    # plt.imshow(large_mask, cmap='jet')

    # plt.show()

    # save the results in a folder results 

    cv2.imwrite(join('results', image_path), large_image)
    cv2.imwrite(join('results', mask_path), large_mask)
    cv2.imwrite(join('results', 'pred_'+mask_path), rgb_reconstructed_classes_large)





