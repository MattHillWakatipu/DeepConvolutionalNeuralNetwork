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


def evaluate(test_generator):
    """
    Evaluation on test images.
    :param test_generator: A testing data generator.
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model('model/model.h5')
    print(model.summary())
    return model.evaluate(test_generator, batch_size=batch_size, verbose=1)


def create_test_generator(directory):
    """
    Initialize a data generator for testing from file.
    :return: The testing data generator
    """
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=directory,
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
    image_size = (300, 300)

    # Load images
    test_generator = create_test_generator(test_data_dir)

    # Evaluation
    loss_and_metrics = evaluate(test_generator)
    print("Test loss:{}\nTest accuracy:{}".format(loss_and_metrics[0], loss_and_metrics[1]))
