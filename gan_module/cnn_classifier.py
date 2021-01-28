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
    model.add(layers.Conv2D(16, (1, 2), padding='same', activation='relu', input_shape=(1, blocksize, 2)))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.Reshape((1, blocksize, 64)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(20))
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
        to_numpy().reshape(sample_size, 1, blockSize, 2)
    test_features = test_dataset.loc[:, ['fake_signal_real', 'fake_signal_imag']].\
        to_numpy().reshape(test_size, 1, blockSize, 2)
    train_labels = train_dataset.labels.to_numpy().reshape(sample_size, blockSize)
    test_labels = test_dataset.labels.to_numpy().reshape(test_size, blockSize)
    classifier = cnn_classifier(blockSize)
    classifier.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = classifier.fit(train_features, train_labels, epochs=100,batch_size=1,
                        validation_data=(test_features, test_labels))
    return history