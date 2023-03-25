from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random


def grabcut_algorithm(original_image, bounding_box, draw):
    '''Apply the GrabCut algorithm to the image.'''

    segment = np.zeros(original_image.shape[:2], np.uint8)
    x, y, width, height = bounding_box
    segment[y:y+height, x:x+width] = 1

    background_mdl = np.zeros((1, 65), np.float64)
    foreground_mdl = np.zeros((1, 65), np.float64)

    cv2.grabCut(original_image, segment, bounding_box,
                background_mdl, foreground_mdl, 5, cv2.GC_INIT_WITH_RECT)

    new_mask = np.where((segment == 2) | (segment == 0), 0, 1).astype('uint8')
    masked_image = original_image*new_mask[:, :, np.newaxis]

    # Set background to bright green (RGB color: 0, 255, 0)
    background_mask = np.where((new_mask == 0), 255, 0).astype('uint8')
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2RGB)
    background_mask[:, :, 0:2] = 0
    masked_image = cv2.addWeighted(masked_image, 1, background_mask, 1, 0)

    if draw:
        cv2.imshow('Result', masked_image)
        cv2.waitKey(0)
    return background_mask


def display_image(file):

    # The function cv2.imread() is used to read an image.
    img_grayscale = cv2.imread(file)

    img_grayscale = resize_with_aspect_ratio(
        img_grayscale, width=500)  # Resize by width OR

    # The function cv2.imshow() is used to display an image in a window.
    cv2.imshow('graycsale image', img_grayscale)

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv2.waitKey(0)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


class PCB:
    '''Load PCB from path'''

    def __init__(self, pcb_path):
        '''Load RGB, mask and annotation from path'''
        self.RGB = pcb_path+'.jpg'
        self.mask = pcb_path+'-mask.png'
        self.ano = pcb_path+'-annot.txt'
        self.path = pcb_path

    def show_rgb(self):
        display_image(self.RGB)

    def load_image(self, type):

        if type == 'rgb':
            img = cv2.imread(self.RGB)
        if type == 'mask':
            img = cv2.imread(self.mask)
        return img

    def load_annotation(self):
        '''Load annotation from path as list of tubles'''
        with open(self.ano, 'r') as f:
            content = f.readlines()  # read file as list of lines
            # remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            # split each line into list of strings
            content = [x.split(' ') for x in content]
            content = [(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(
                x[4])) for x in content]  # convert list of strings to list of tubles
        return content

    def load_annotation_on_rgb(self):
        '''Load annotation drawn on image'''
        # load rgb image
        img = cv2.imread(self.RGB)
        # load annotation
        annotation = self.load_annotation()
        # draw a rotated bounding box on image
        for i in range(len(annotation)):
            x, y, w, h, angle = annotation[i]
            box = cv2.boxPoints(((x, y), (w, h), angle))
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 20)
        return img

    def load_annotation_on_mask(self):
        '''Load annotation drawn on mask as green patch'''
        # load mask image
        img = cv2.imread(self.mask)
        # load annotation
        annotation = self.load_annotation()
        # draw a rotated bounding box on image
        for i in range(len(annotation)):
            x, y, w, h, angle = annotation[i]
            box = cv2.boxPoints(((x, y), (w, h), angle))
            box = np.int0(box)
            patch = np.array([box[0], box[1], box[2], box[3]], dtype=np.int32)
            cv2.fillPoly(img, [patch], (0, 255, 0))
        return img


class DATASET:
    '''Load dataset from path_dataset'''

    def __init__(self, path_dataset):  # path_dataset = 'dataset'
        self.pcb_path = []
        for subset in sorted(listdir(path_dataset)):
            if not isfile(subset):
                # print('load subset: {}'.format(subset))
                if not isfile(subset):
                    for pcbn in sorted(listdir(join(path_dataset, subset))):
                        if not isfile(pcbn):
                            print(' load pcb: {}'.format(pcbn[3:]))
                            image = [image[:4]for image in listdir(
                                join(path_dataset, subset, pcbn))]

                            for instance in sorted(set(image)):
                                # print(instance)
                                self.pcb_path.append(
                                    join(path_dataset, subset, pcbn, instance))

    def load(self, n):
        pcb = PCB(self.pcb_path[n])
        return pcb


def plot_rgb_mask(rgb, mask):
    # Create two windows
    cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
    cv2.namedWindow('MASK', cv2.WINDOW_NORMAL)

    # Move windows next to each other
    cv2.moveWindow('RGB', 0, 0)
    cv2.moveWindow('MASK', rgb.shape[1], 0)

    # Display images in windows
    cv2.imshow('RGB', np.asarray(rgb))
    cv2.imshow('MASK', np.asarray(mask))

    # Wait for 1 second
    cv2.waitKey(20)

    # Close windows
    cv2.destroyAllWindows()


def cut_patches(image, patch_size):
    patches = []

    # Calculate the number of patches in each dimension
    num_patches_x = image.shape[1] // patch_size
    num_patches_y = image.shape[0] // patch_size

    # Iterate over each patch
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # Calculate the start and end indices for the patch
            start_x = x * patch_size
            end_x = start_x + patch_size
            start_y = y * patch_size
            end_y = start_y + patch_size

            # Cut the patch
            patch = image[start_y:end_y, start_x:end_x]
            patches.append(patch)

    return patches


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


# def load_image_train(pcb):
#     input_image = pcb["rgb"]
#     input_mask = pcb["mask"]
#     input_image, input_mask = resize(input_image, input_mask)
#     input_image, input_mask = augment(input_image, input_mask)
#     input_image, input_mask = normalize(input_image, input_mask)

#     return input_image, input_mask


# def load_image_test(pcb):
#     input_image = pcb["image"]
#     input_mask = pcb["segmentation_mask"]
#     input_image, input_mask = resize(input_image, input_mask)
#     input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def copy_to_dataset(path_dataset):
    ''' Copy the images, masks, anotated masks and anonations (as txt) to a new folder to be used by the model '''

    dataset = DATASET(path_dataset)

    for i in range(1, len(dataset.pcb_path)+1):
        pcb = dataset.load(i)

        rgb = pcb.load_image('rgb')
        mask = pcb.load_image('mask')
        icmask = pcb.load_annotation_on_mask()
        path = pcb.path.split('/')
        anno = pcb.load_annotation()

        print('Saving image {} of {}'.format(i, len(dataset.pcb_path)))

        cv2.imwrite(join(path_dataset, 'pcb_dataset',  'image',
        path[1]+'-'+path[2]+'-'+path[3]+'.jpg'), rgb)

        cv2.imwrite(join(path_dataset, 'pcb_dataset',  'mask',
        path[1]+'-'+path[2]+'-'+path[3]+'.png'), mask)

        cv2.imwrite(join(path_dataset, 'pcb_dataset', 'icmask',
                        path[1]+'-'+path[2]+'-'+path[3]+'.png'), icmask)

        with open(join(path_dataset, 'pcb_dataset', 'annotation',
                       path[1]+'-'+path[2]+'-'+path[3]+'.txt'), 'w') as f:
            for i in range(len(anno)):
                f.write('{} {} {} {} {}\n'.format(
                    anno[i][0], anno[i][1], anno[i][2], anno[i][3], anno[i][4]))
            f.close()


def move_files_to_test_folders(path_dataset):
    # Get a list of all the image filenames
    image_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(
        join(path_dataset, 'image')) if filename.endswith(".jpg")]

    # Select 20 random filenames
    random.seed(42)  # For reproducibility
    filenames_to_move = random.sample(image_filenames, 20)

    # Move the selected filenames to their corresponding _test folders
    for filename in filenames_to_move:
        annotation_src_path = os.path.join(
            path_dataset, "annotation", filename + ".txt")
        annotation_dst_path = os.path.join(path_dataset,
                                           "annotation_test", filename + ".txt")
        os.rename(annotation_src_path, annotation_dst_path)

        icmask_src_path = os.path.join(
            path_dataset, "icmask", filename + ".png")
        icmask_dst_path = os.path.join(
            path_dataset, "icmask_test", filename + ".png")
        os.rename(icmask_src_path, icmask_dst_path)

        mask_src_path = os.path.join(
            path_dataset, "mask", filename + ".png")
        mask_dst_path = os.path.join(
            path_dataset, "mask_test", filename + ".png")
        os.rename(mask_src_path, mask_dst_path)

        image_src_path = os.path.join(path_dataset, "image", filename + ".jpg")
        image_dst_path = os.path.join(
            path_dataset, "image_test", filename + ".jpg")
        os.rename(image_src_path, image_dst_path)


def main():

    return


if __name__ == "__main__":

    dataset = DATASET('dataset')

    # create test and train folders

    #copy_to_dataset('dataset')

    move_files_to_test_folders('dataset/pcb_dataset')


    