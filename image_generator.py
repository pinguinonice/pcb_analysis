
#from tifffile import imsave, imread
from matplotlib import pyplot as plt
import random
import os
import numpy as np
import cv2
from patchify import patchify, unpatchify
from keras.utils import to_categorical


def load_img(img_dir, img_name, image_size):
    images = []

    if (img_name.split('.')[1] == 'jpg') or (img_name.split('.')[1] == 'png'):

        image = cv2.imread(os.path.join(img_dir, img_name))
        image = cv2.resize(image, image_size,
                           interpolation=cv2.INTER_NEAREST)

    image = np.array(image)

    return(image)


def make_patches(image, size_x, size_y):
    image_patches = patchify(
        image, (size_x, size_y, image.shape[-1]), step=int(size_y))
    r = image_patches.shape[0]
    c = image_patches.shape[1]
    all_images = np.empty((r*c, size_x, size_y, image.shape[-1]))
    n = 0
    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):

            patch = image_patches[i, j, :, :]
            patch = patch.squeeze(axis=0)

            all_images[n] = patch
            n += 1
    return all_images


def get_random_image_and_mask_patches(L, image_size, img_dir, img_list, mask_dir, mask_list, size_x, size_y, classes):

    # check if jpg other wise get other file
    while True:
        random_image = random_int = random.randint(0, L-1)
        if 'jpg' or 'png' in img_list[random_image].split('.')[-1]:
            break

    #print('load image {}'.format(img_list[random_image]))

    X = load_img(img_dir, img_list[random_image], image_size)
    Y = load_img(mask_dir, mask_list[random_image], image_size)
    Y = np.where(Y > 128, 1, 0)

    # Create an empty one-hot encoded array with the same dimensions as Y
    n_classes = len(classes)
    Y_one_hot = np.zeros((Y.shape[0], Y.shape[1], n_classes), dtype=np.uint8)

    # Fill in the one-hot encoded array based on the class of each pixel in Y
    for i, class_color in enumerate(classes.values()):
        mask = np.all(Y == class_color, axis=2)
        Y_one_hot[mask, i] = 1
    Y = Y_one_hot
    X_patches = make_patches(X, size_x, size_y)
    Y_patches = make_patches(Y, size_x, size_y)

    return X_patches, Y_patches


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, image_size, n_input_band, n_classes, size_x, size_y, classes):

    L = len(img_list)

    X = np.empty((batch_size, image_size[0], image_size[1], n_input_band))
    Y = np.empty((batch_size, image_size[0], image_size[1], n_classes))

    # keras needs the generator infinite, so we will use while true
    while True:

        # draw 2 images with each r*c pathces
        patches_X1, patches_Y1 = get_random_image_and_mask_patches(
            L, image_size, img_dir, img_list, mask_dir, mask_list, size_x, size_y, classes)
        patches_X2, patches_Y2 = get_random_image_and_mask_patches(
            L, image_size, img_dir, img_list, mask_dir, mask_list, size_x, size_y, classes)

        combined_X = np.concatenate([patches_X1, patches_X2], axis=0)
        combined_Y = np.concatenate([patches_Y1, patches_Y2], axis=0)

        perm = np.random.permutation(combined_X.shape[0])

        # Shuffle both arrays along the first dimension
        combined_X = combined_X[perm]
        combined_Y = combined_Y[perm]

        # draw image patches untill the combined_array is longer than batch_size
        while combined_X.shape[0] < batch_size:
            patches_Xn, patches_Yn = get_random_image_and_mask_patches(
                L, image_size, img_dir, img_list, mask_dir, mask_list, size_x, size_y, classes)

            combined_X = np.concatenate([combined_X, patches_Xn], axis=0)
            combined_Y = np.concatenate([combined_Y, patches_Yn], axis=0)

            perm = np.random.permutation(combined_X.shape[0])

            # Shuffle both arrays along the first dimension
            combined_X = combined_X[perm]
            combined_Y = combined_Y[perm]

        X = combined_X[:batch_size].astype(np.float32)/255.
        Y = combined_Y[:batch_size].astype(np.float32)

        yield (X, Y)


if __name__ == "__main__":

    ###########################################
    # Test the generator
    train_img_dir = "../dataset/pcb_dataset/image"
    train_mask_dir = "../dataset/pcb_dataset/icmask"
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

    batch_size = 64
    SIZE_X, SIZE_Y = 256, 256
    r, c = 3, 5
    image_size = (r*SIZE_X, c*SIZE_Y)

    n_input_band = 3
    classes = {'bg': [0, 0, 0], 'ic': [0, 1, 0], 'pcb': [1, 1, 1]}
    n_classes = len(classes)

    train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                    train_mask_dir, train_mask_list, batch_size, image_size, n_input_band, n_classes, SIZE_X, SIZE_Y, classes)

    while True:
        img, msk = train_img_datagen.__next__()

        # display random 6 images and masks in the batch next to each other
        for i in range(6):
            img_num = random.randint(0, img.shape[0]-1)
            test_img = img[img_num]
            test_mask = msk[img_num]
            #test_mask = test_mask[:, :, 1]

            plt.figure(figsize=(12, 8))

            plt.subplot(121)
            plt.imshow(test_img)
            plt.title('Image')
            plt.subplot(122)
            plt.imshow(test_mask)
            plt.title('Mask')
            plt.show()

        # img_num = random.randint(0, img.shape[0]-1)
        # test_img = img[img_num]
        # test_mask = msk[img_num]
        # #test_mask = test_mask[:, :, 1]

        # plt.figure(figsize=(12, 8))

        # plt.subplot(121)
        # plt.imshow(test_img)
        # plt.title('Image')
        # plt.subplot(122)
        # plt.imshow(test_mask)
        # plt.title('Mask')
        # plt.show()
