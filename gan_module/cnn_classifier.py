import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import mlcTest as mlc



def cnn_classifier(blocksize):
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (1, 2), padding='same', activation='relu', input_shape=(blocksize, 1, 2)))
    model.add(layers.MaxPool2D(1,5))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPool2D(1,5))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    return model



def mlp_classifier(blocksize):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(1, blocksize, 2)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ])
    return model

'''
def cnn_classifier(blockSize):
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (1, 1), activation='relu', input_shape=(1, blockSize, 2)))
    model.add(layers.MaxPooling2D(1,1))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(1,1))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(1,1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(20))
    return model
'''


def training(data, blockSize):
    block = int(data.shape[0]/blockSize)
    sample_size, train_dataset = mlc.training_set(data, block)
    test_size = block - sample_size
    test_dataset = mlc.test_set(data, train_dataset)
    train_features = train_dataset.loc[:, ['fake_signal_real', 'fake_signal_imag']].\
        to_numpy().reshape(sample_size, blockSize, 1, 2)
    test_features = test_dataset.loc[:, ['fake_signal_real', 'fake_signal_imag']].\
        to_numpy().reshape(test_size, blockSize, 1, 2)
    train_labels = train_dataset.groupby('block').mean().labels
    test_labels = test_dataset.groupby('block').mean().labels

    print(test_labels)
    classifier = cnn_classifier(blockSize)
    classifier.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = classifier.fit(train_features, train_labels, epochs=100,
                        validation_data=(test_features, test_labels))
    return history