#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf
import random

from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten

from test import load_images, convert_img_to_array, preprocess_data

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()

    # 3 Convolutional layers with ReLU activation followed by max-pooling
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['categorical_accuracy'])

    print(model.summary())

    return model


def load_training_data():
    # Test folder
    train_data_dir = "data/train"

    # Image size, please define according to your settings when training your model.
    image_size = (128, 128)

    # Load images
    images, labels = load_images(train_data_dir, image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    x_train, y_train = convert_img_to_array(images, labels)

    # Preprocess data.
    x_train = preprocess_data(x_train)

    # Exploratory Data Analysis
    print("Training data loaded")
    print("Instances: {}".format(x_train.shape[0]))
    print("Size: {} x {} x {}".format(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    return x_train, y_train


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    x_train, y_train = load_training_data()
    # TODO validation_split=0.1
    model.fit(x_train, y_train,
              batch_size=256,
              epochs=50,
              verbose=1)
    loss_and_metrics = model.evaluate(x_train, y_train, verbose=0)
    print("Test loss:{}\nTest accuracy:{}".format(loss_and_metrics[0], loss_and_metrics[1]))
    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    model.save("model/model.h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':
    model = construct_model()
    model = train_model(model)
    save_model(model)
