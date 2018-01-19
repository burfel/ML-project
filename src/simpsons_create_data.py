"""
The dataset was taken from:
https://www.kaggle.com/alexattia/the-simpsons-characters-dataset

It must be stored some place on your device. The path to this folder is the 
"base_path" in this program. This base_path must be set manually. Executing this
file will create the features we used to solve this task (HOG features 
concatenated with mean value patches of the images). 
"""

import os
import re

from Decision_Tree import DecisionTree
from Random_Forest import RandomForest

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from skimage.feature import hog

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.util import view_as_blocks


"""
Visualizes several images of Simpsons characters with their respective names.
Takes:
    x: array of many input images of shape: (nr_imgs, height, width, 3)
    y: array of labels (nr_imgs)
    character_names: dictionary of key=label, value=name corresponding to label
    n: number of images in x to be displayed
"""
def show_images(x, y, character_names, n):
    for i in range(n):
        plt.imshow(x[i])
        plt.title('class: '+character_names[y[i]])
        plt.show()
    


def shuffle_data(x, y):
    #Shuffles the images and the labels by some random (but same) permutation
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]

def get_character_names(path):
    # reads the names out of the folder paths
    folders = [folder for folder in os.listdir(path) if folder!='.DS_Store']
    return folders


def load_train_simpsons(path, debug=False):
    """
    loads the simpsons training data from a path and returns it as an array
    Takes:
        path: path to the data folder with training data folders
        debug: takes a boolean, True greatly reduces the number of images loaded
    """
    folders = [folder for folder in os.listdir(path) if folder!='.DS_Store']
    print 'folders: ', folders, '\n'
    x, y = [], []
    for i in range(len(folders)):
        print 'folder ', i+1, ' of ', len(folders), folders[i]
        image_names = os.listdir(os.path.join(path, folders[i]))
        n = len(image_names) if not debug else min(len(image_names), 10)
        image_paths = [os.path.join(path, folders[i], image_names[j]) for j in range(n)]
        x.extend([imread(image_paths[j]) for j in range(len(image_paths))])
        y.extend([i for bla in image_paths])
    return np.array(x), np.array(y), folders


def load_test_simpsons(path, character_names, debug=False):
    """
    loads the simpsons test data from a path and returns it as an array
    Takes:
        path: path to the data folder with testing data
        character_names: Dictionary corresponding to the character names given labels
        debug: takes a boolean, True greatly reduces the number of images loaded    
    """
    x, y = [], []
    image_names = [img for img in os.listdir(path) if img.endswith('.jpg')]
    # get character names from the image paths
    characters_for_lookup = [image_names[i][0:[m.start() for m in re.finditer('_', image_names[i])][-1]] for i in range(len(image_names))]
    if debug: print image_names[:20], characters_for_lookup[:20]
    n = len(image_names) if not debug else min(len(image_names), 100)
    image_paths = [os.path.join(path, image_names[i]) for i in range(n)]
    x = [imread(image_paths[i]) for i in range(len(image_paths))]
    y = [character_names.index(characters_for_lookup[i]) for i in range(len(characters_for_lookup))][0:n]
    return np.array(x), np.array(y)


def preprocessing(x, y, shuffle=True, resize=True, resize_shape=(110,110), extract_hog=True, mean_patches=True, pixels_per_cell=(10,10), flip=True, shift=True, shift_max=(4,4)):
    """
    A function which contains all the preprocessing functions we deemed 
    potentially useful. Turned out to be complete overkill.
    Takes:
        x: images, an ndarray of shape (nr_imgs, height, width, 3)
        y: labels, an array of shape (nr_imgs)
        shuffle: boolean that decides whether data is shuffled or not
        resizing: tuple that resizes the images to the shape of the tuple
        extract_hog: boolean which decides whether the hog_features are computed
        mean_patches: boolean which decides whether the mean patches are concatenated 
                        to the hog_features
        pixels_per_cell: tuple which decides the cell size for the hog descriptors 
                            and the patch size of mean_patches
        flip: boolean, decides whether random flips are applied to the data (data 
                inflation technique)
        shift: boolean, decides whether random shifts are applied to the image (data 
                inflation technique)
        shift_max: tuple, determines the maximum shifts applied in y- and x-direction
    Returns: 
        transformed data
    """
    if shuffle:
        x, y = shuffle_data(x, y)
    if resize:
        x = np.array([imresize(img, resize_shape+(3,)) for img in x])
    if flip:
        for i in range(len(x)):
            if np.random.random()>0.5:
                x[i] = x[i,:,::-1,:]
    if shift:
        for i in range(len(x)):
            down_shift  = int(np.random.random()*shift_max[0])
            right_shift = int(np.random.random()*shift_max[1])
            x[i] = np.roll(x[i], down_shift,  axis=0)
            x[i] = np.roll(x[i], right_shift, axis=1)
    original_x, original_y = x, y
    if extract_hog:
        new_x = []
        for i in range(len(x)):
            if i%10==0: print i, " of ", len(x)
            grey_image = color.rgb2gray(x[i])
            fd, hog_img = hog(grey_image, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1,1), visualise=True)
            if mean_patches:
                mean_patch_colors = view_as_blocks(x[i], pixels_per_cell+(3,)).mean(axis=-2).mean(axis=-2).flatten()
                new_x.append(np.concatenate((mean_patch_colors, fd.flatten()), axis=0))
            else: new_x.append(fd.flatten())
        x = np.array(new_x)
    return x, y, original_x, original_y





if __name__=='__main__':

    # the path of the-simpsons-characters-dataset folder on your machine
    base_path = '/home/thomas/Desktop/simpsons_work/the-simpsons-characters-dataset'


    if not os.path.exists(os.path.join(base_path, 'simpsons_dataset_features')):
        os.makedirs(os.path.join(base_path, 'simpsons_dataset_features'))

    """
    load simpsons train data
    tranform it to feature space
    store it
    """
    train_path = os.path.join(base_path,'simpsons_dataset')
    x_train, y_train, character_names = load_train_simpsons(train_path, debug=False)
    show_images(x_train, y_train, character_names, 3)
    feature_train, label_train, x_train, y_train = preprocessing(x_train, y_train, shuffle=True, resize=True, resize_shape=(20*7,20*7), extract_hog=True, mean_patches=True, pixels_per_cell=(7,7), flip=False, shift=False, shift_max=(7,7))
    print 'x and y train shapes: ', feature_train.shape, label_train.shape


    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'train_feature_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), feature_train)
    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'train_label_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), label_train)
    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'train_x_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), x_train)
    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'train_y_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), y_train)


    """
    load simpsons test data
    tranform it to feature space
    store it
    """
    character_names = get_character_names(os.path.join(base_path, 'simpsons_dataset'))
    test_path = os.path.join(base_path,'kaggle_simpson_testset')
    x_test, y_test = load_test_simpsons(test_path, character_names, debug=False)
    show_images(x_test, y_test, character_names, 1)
    feature_test, label_test, x_test, y_test = preprocessing(x_test, y_test, shuffle=False, resize=True, resize_shape=(20*7,20*7), extract_hog=True, pixels_per_cell=(7,7), flip=False, shift=False, shift_max=(7,7))
    print 'x and y test shapes: ', feature_test.shape, label_test.shape



    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'test_feature_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), feature_test)
    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'test_label_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), label_test)
    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'test_x_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), x_test)
    np.save(os.path.join(base_path, 'simpsons_dataset_features', 'test_y_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'), y_test)






















