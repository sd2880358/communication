import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import random

import cycleGAN as cycle

class Denoise(Model):
    def __init__(self, blockSize):
        super(Denoise, self).__init__()
        self.blockSize = blockSize
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(blockSize, 2, 1)),
            layers.Conv2D(16, (5, 2), strides=(5, 2), activation="relu", padding='same'),
            layers.AveragePooling2D((1, 1)),
            layers.Conv2D(8, (5, 1), strides=(5, 1), activation="relu", padding='same'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=(5, 1), strides=(5, 1), activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=(5, 2), strides=(5, 2), activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same')
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




def train(data, blockSize, date):
    autoencoder = Denoise(blockSize)
    test_feature, test_label, symbol, my_table = cycle.shuffle_data(data, blockSize)
    print(test_feature.shape)
    autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
    history = autoencoder.fit(test_feature, test_label, epochs=100, verbose=0)
    hist = pd.DataFrame(history.history)
    hist.to_csv("./result/" + date + "/result")
    return history


if __name__ == '__main__':
    date = "2_1/"
    file_name = "my_data1"
    data_label = "my_labels1"
    data = cycle.dataset(file_name, data_label)
    train(data, 50, date)