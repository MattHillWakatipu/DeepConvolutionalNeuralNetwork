#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
print(tf.version.VERSION)


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default="data/test",
                      help="path to test_data_dir")
    args = vars(args.parse_args())
    return args


def load_images(directory, image_size=(300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    image_paths = list(paths.list_images(directory))
    for image_path in image_paths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)
    return images_data, sorted(labels)


def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    x_test = np.array(images, dtype="float") / 255.0
    y_test = np.array(labels)

    # Binarize the labels
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return x_test, y_test


def preprocess_data(x):
    """
    Pre-process the test data.
    :param x: the original data
    :return: the preprocess data
    """
    # NOTE: # If you have conducted any pre-processing on the image,
    # please implement this function to apply onto test images.
    return x


def evaluate(x_test, y_test):
    """
    Evaluation on test images
    ******Please do not change this function******
    :param x_test: test images
    :param y_test: test labels
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model('model/smalltrain.h5')
    print(model.summary())
    return model.evaluate(x_test, y_test, batch_size, verbose=1)


def gen_evaluate(test_generator):
    """
    Evaluation on test images.
    :param test_generator: A testing data generator.
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model('model/smalltrain.h5')
    print(model.summary())
    return model.evaluate(test_generator, batch_size=batch_size, verbose=0)


def create_test_generator():
    """
    Initialize a data generator for testing from file.
    :return: The testing data generator
    """
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        directory='data/test',
        target_size=image_size,
        classes=['cherry', 'strawberry', 'tomato'],
        batch_size=16,
        shuffle=False)
    return test_generator


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["test_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    # Load images
    images, labels = load_images(test_data_dir, image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    x_test, y_test = convert_img_to_array(images, labels)

    # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    x_test = preprocess_data(x_test)

    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    # FIXME need to pre process this somehow? Generator based testing works correctly but this does not
    loss, accuracy = evaluate(x_test, y_test)
    print("loss={}, accuracy={}".format(loss, accuracy))

    # Valuation using generator
    test_generator = create_test_generator()
    loss_and_metrics = gen_evaluate(test_generator)
    print("Test loss:{}\nTest accuracy:{}".format(loss_and_metrics[0], loss_and_metrics[1]))
