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

import numpy as np
import tensorflow as tf
import random

from tensorflow.python.distribute.multi_process_lib import multiprocessing
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten

# Set random seeds to ensure the reproducible results
SEED = 309
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

batch_size = 64
image_size = (300, 300)
training_size = 3600
validation_size = 450
testing_size = 150


def create_data_generators():
    """
    Create the data generators from file for each of the datasets.
    :return: Created Data_Generators
    """
    class_tuple = ['cherry', 'strawberry', 'tomato']

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        directory='data/train',
        target_size=image_size,
        classes=class_tuple,
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        directory='data/validation',
        target_size=image_size,
        classes=class_tuple,
        batch_size=batch_size)

    test_generator = test_datagen.flow_from_directory(
        directory='data/test',
        target_size=image_size,
        classes=class_tuple,
        batch_size=batch_size,
        shuffle=False)

    assert train_generator.n == training_size
    assert validation_generator.n == validation_size
    assert test_generator.n == testing_size
    assert train_generator.num_classes == validation_generator.num_classes == test_generator.num_classes == 3

    return train_generator, validation_generator, test_generator


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

    # Block 1 convolution layer
    model.add(Conv2D(input_shape=(300, 300, 3), filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2 convolution layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3 convolution layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4 convolution layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected classifier using softmax
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


def construct_mlp_model():
    """
    Construct the MLP model.
    :return: model: the initial MLP model.
    """
    model = Sequential()

    model.add(Input(shape=(300, 300, 3)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


def train_model(model, train_generator, validation_generator):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=10, verbose=1),
        ModelCheckpoint(filepath='./model/checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True),
        TensorBoard(log_dir='./logs')
    ]

    model.fit(train_generator,
              epochs=200,
              steps_per_epoch=training_size // batch_size,
              validation_data=validation_generator,
              validation_steps=validation_size // batch_size,
              callbacks=callbacks,
              workers=multiprocessing.cpu_count(),
              max_queue_size=512,
              verbose=2)

    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    model.save("model/model.h5")
    print("Model Saved Successfully.")


def split_data():
    """
    Split the data into training, validation, and testing sets.
    :return:
    """
    class_training_size = training_size // 3
    class_validation_size = validation_size // 3
    class_testing_size = testing_size // 3

    if os.path.isdir('data/train/cherry') is False:
        os.chdir('data')

        os.makedirs('train/cherry')
        os.makedirs('train/strawberry')
        os.makedirs('train/tomato')

        os.makedirs('validation/cherry')
        os.makedirs('validation/strawberry')
        os.makedirs('validation/tomato')

        os.makedirs('test/cherry')
        os.makedirs('test/strawberry')
        os.makedirs('test/tomato')

        for c in random.sample(glob.glob('cherry*'), class_training_size):
            shutil.move(c, 'train/cherry')
        for c in random.sample(glob.glob('strawberry*'), class_training_size):
            shutil.move(c, 'train/strawberry')
        for c in random.sample(glob.glob('tomato*'), class_training_size):
            shutil.move(c, 'train/tomato')

        for c in random.sample(glob.glob('cherry*'), class_validation_size):
            shutil.move(c, 'validation/cherry')
        for c in random.sample(glob.glob('strawberry*'), class_validation_size):
            shutil.move(c, 'validation/strawberry')
        for c in random.sample(glob.glob('tomato*'), class_validation_size):
            shutil.move(c, 'validation/tomato')

        for c in random.sample(glob.glob('cherry*'), class_testing_size):
            shutil.move(c, 'test/cherry')
        for c in random.sample(glob.glob('strawberry*'), class_testing_size):
            shutil.move(c, 'test/strawberry')
        for c in random.sample(glob.glob('tomato*'), class_testing_size):
            shutil.move(c, 'test/tomato')

        os.chdir('..')


def test_model(model):
    print("Testing model...")
    loss_and_metrics = model.evaluate(train_generator, verbose=0)
    print("Train loss:{}\nTrain accuracy:{}\n".format(loss_and_metrics[0], loss_and_metrics[1]))
    loss_and_metrics = model.evaluate(validation_generator, verbose=0)
    print("Validation loss:{}\nValidation accuracy:{}\n".format(loss_and_metrics[0], loss_and_metrics[1]))
    loss_and_metrics = model.evaluate(test_generator, verbose=0)
    print("Test loss:{}\nTest accuracy:{}\n".format(loss_and_metrics[0], loss_and_metrics[1]))


if __name__ == '__main__':
    # Split the dataset into smaller size, may remove later
    split_data()

    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators()

    # Construct the model
    model = construct_model()

    # Train the model
    model = train_model(model, train_generator, test_generator)

    # Test the model
    test_model(model)

    # Save the model
    save_model(model)
