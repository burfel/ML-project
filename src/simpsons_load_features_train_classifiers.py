"""
Loads the features and the labels, trains classifiers on the data and shows 
the accuracy.
"""


import os
import numpy as np

import simpsons_create_data

from Decision_Tree import DecisionTree
from Random_Decision_Tree import RandomDecisionTree
from Random_Forest import RandomForest

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog



# the path of the-simpsons-characters-dataset folder on your machine
base_path = '/home/thomas/Desktop/simpsons_work/the-simpsons-characters-dataset'
character_names = simpsons_create_data.get_character_names(os.path.join(base_path, 'simpsons_dataset'))

"""
load training data
"""
print "loading train data...", "\n"
feature_train, label_train, x_train, y_train = np.load(os.path.join(base_path, 'simpsons_dataset_features', 'train_feature_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy')), np.load(os.path.join(base_path, 'simpsons_dataset_features', 'train_label_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy')), np.load(os.path.join(base_path, 'simpsons_dataset_features', 'train_x_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy')), np.load(os.path.join(base_path, 'simpsons_dataset_features', 'train_y_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'))



"""
load testing data
"""
print "loading test data...", "\n"
feature_test, label_test, x_test, y_test = np.load(os.path.join(base_path, 'simpsons_dataset_features', 'test_feature_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy')), np.load(os.path.join(base_path, 'simpsons_dataset_features', 'test_label_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy')), np.load(os.path.join(base_path, 'simpsons_dataset_features', 'test_x_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy')), np.load(os.path.join(base_path, 'simpsons_dataset_features', 'test_y_shuffle_resize_140x140_extract_hog_mean_patches_pixels_per_cell_7x7_flip_shift_7x7.npy'))



"""
train classifier
"""
print "training classifier..."
n_examples = len(feature_train)
n_estimators = 5
max_depth = 25
print "\tnr estimators: ", n_estimators
print "\tmax depth: ", max_depth, "\n"
#classifier = ExtremelyRandomForest(nr_trees=n_estimators, maximum_depth=max_depth, min_examples_per_leaf=20)
classifier = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=2)
#classifier = RandomForest(nr_trees=n_estimators, maximum_depth=max_depth, min_examples_per_leaf=20)
#classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=20)
classifier.fit(feature_train[0:n_examples], label_train[0:n_examples])


# this is of course affected by overfitting but it's still an interesting value
print "predicting labels from train data..."
predicted = classifier.predict(feature_train)
#predicted = np.array([classifier.classify(feature_train[i]) for i in range(len(feature_train))])
correct_prediction = [label_train[i]==predicted[i] for i in range(len(feature_train))]

print sum(correct_prediction), " of ", len(label_train)
print sum(correct_prediction)*1.0 / len(label_train), "\n"



print "predicting labels from test data..."
predicted = classifier.predict(feature_test)
#predicted = np.array([classifier.classify(feature_test[i]) for i in range(len(feature_test))])
correct_prediction = [label_test[i]==predicted[i] for i in range(len(feature_test))]

print sum(label_test==predicted), " of ", len(label_test)
print sum(label_test==predicted)*1.0 / len(label_test), "\n"





