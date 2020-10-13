#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py"
    so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
import glob
import os
import shutil

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
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

    # 3 x Convolutional layers with ReLU activation followed by max-pooling
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(300, 300, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    # model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(300, 300, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(3))
    # model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def create_data_generators():
    generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    image_size = (300, 300)
    class_tuple = ['cherry', 'strawberry', 'tomato']

    train_generator = train_datagen.flow_from_directory(
        directory='train',
        target_size=image_size,
        classes=class_tuple,
        batch_size=10)

    validation_generator = test_datagen.flow_from_directory(
        directory='validation',
        target_size=image_size,
        classes=class_tuple,
        batch_size=10)

    test_generator = generator.flow_from_directory(
        directory='test',
        target_size=image_size,
        classes=class_tuple,
        batch_size=10,
        shuffle=False)

    assert train_generator.n == 3600
    assert validation_generator.n == 450
    assert test_generator.n == 450
    assert train_generator.num_classes == validation_generator.num_classes == test_generator.num_classes == 3

    return train_generator, validation_generator, test_generator


def train_model(model, train_generator, validation_generator,):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    model.fit(train_generator,
              steps_per_epoch=360,
              epochs=5,
              validation_data=validation_generator,
              validation_steps=45,
              verbose=1)

    loss_and_metrics = model.evaluate(test_generator, verbose=0)
    print("Test loss:{}\nTest accuracy:{}".format(loss_and_metrics[0], loss_and_metrics[1]))
    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    print(os.getcwd())
    os.chdir('..')
    model.save("model/model.h5")
    print("Model Saved Successfully.")


def split_data():
    """
    Split the data into training, validation, and testing sets.
    :return:
    """
    os.chdir('data')

    if os.path.isdir('train/cherry') is False:
        os.makedirs('train/cherry')
        os.makedirs('train/strawberry')
        os.makedirs('train/tomato')

        os.makedirs('validation/cherry')
        os.makedirs('validation/strawberry')
        os.makedirs('validation/tomato')

        os.makedirs('test/cherry')
        os.makedirs('test/strawberry')
        os.makedirs('test/tomato')

        for c in random.sample(glob.glob('cherry*'), 1200):
            shutil.move(c, 'train/cherry')
        for c in random.sample(glob.glob('strawberry*'), 1200):
            shutil.move(c, 'train/strawberry')
        for c in random.sample(glob.glob('tomato*'), 1200):
            shutil.move(c, 'train/tomato')

        for c in random.sample(glob.glob('cherry*'), 150):
            shutil.move(c, 'validation/cherry')
        for c in random.sample(glob.glob('strawberry*'), 150):
            shutil.move(c, 'validation/strawberry')
        for c in random.sample(glob.glob('tomato*'), 150):
            shutil.move(c, 'validation/tomato')

        for c in random.sample(glob.glob('cherry*'), 150):
            shutil.move(c, 'test/cherry')
        for c in random.sample(glob.glob('strawberry*'), 150):
            shutil.move(c, 'test/strawberry')
        for c in random.sample(glob.glob('tomato*'), 150):
            shutil.move(c, 'test/tomato')


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Split the dataset into smaller size, may remove later
    split_data()

    # Create data generators for
    train_generator, validation_generator, test_generator = create_data_generators()

    # Plot the first 10 images and print their labels
    images, labels = next(train_generator)
    plot_images(images)
    print(labels)

    model = construct_model()
    model = train_model(model, train_generator, test_generator)
    save_model(model)
