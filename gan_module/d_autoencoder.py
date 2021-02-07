import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import random
import mlcTest as mlc
import cnn_classifier as cls

import cycleGAN as cycle

class Denoise(Model):
    def __init__(self, blockSize):
        super(Denoise, self).__init__()
        self.blockSize = blockSize
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(blockSize, 2, 1)),
            layers.Conv2D(16, (5, 2), strides=(2, 2), activation="linear", padding='same'),
            layers.AveragePooling2D((1, 1)),
            layers.Conv2D(8, (5, 1), activation="linear", padding='same'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=(5, 1), activation='linear', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=(5, 2), strides=(2, 2), activation='linear', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same')
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def relative_loss(X, X_pred):
    loss = tf.losses.MeanAbsolutePercentageError()
    return loss(X, X_pred)/100/50



def train(data, blockSize, date, epochs):

    blocks = int(data.shape[0]/blockSize)
    sample_size, train_dataset = mlc.training_set(data, blocks)
    test_size = blocks - sample_size
    test_dataset = mlc.test_set(data, train_dataset)
    train_features = train_dataset.iloc[:, [0,1]].\
        to_numpy().reshape(sample_size, blockSize, 2, 1)
    test_features = test_dataset.iloc[:, [0,1]].\
        to_numpy().reshape(test_size, blockSize, 2, 1)
    train_labels = train_dataset.loc[:, ['label_real', 'label_imag']].\
        to_numpy().reshape(sample_size, blockSize, 2, 1)
    test_labels = test_dataset.loc[:, ['label_real', 'label_imag']].\
        to_numpy().reshape(test_size, blockSize, 2, 1)
    autoencoder = Denoise(blockSize)
    autoencoder.compile(optimizer="adam", loss=losses.MeanAbsoluteError(), metrics=[relative_loss])
    checkpoint_path = "./checkpoints/test4/" + date
    ckpt = tf.train.Checkpoint(autoencoder=autoencoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    history = autoencoder.fit(train_features, train_labels,
                              shuffle=True, epochs=epochs, verbose=0,
                              validation_data=(test_features, test_labels))
    hist = pd.DataFrame(history.history)
    hist.to_csv("./result/" + date + "/losses")
    feature, labels, symbol, my_table = cycle.shuffle_data(data, blockSize)
    prediction = autoencoder.predict(feature)
    result = pd.DataFrame(
        {
            "real": feature.numpy()[:, :, 0].flatten(),
            "imag": feature.numpy()[:, :, 1].flatten(),
            "fake_real": prediction[:, :, 0].flatten(),
            "fake_imag": prediction[:, :, 1].flatten(),
            "block": my_table.block,
            "cons": (my_table.cons.to_numpy() - 1).flatten(),
            "label_real":my_table.label_real,
            "label_imag":my_table.label_imag,
            "labels": (my_table.label.to_numpy()).flatten()}
    )
    result.to_csv("./result/" + date + "result", index=False)
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint  at {}'.format(ckpt_save_path))



if __name__ == '__main__':
    date = "2_4/"
    file_name = "my_data1"
    data_label = "my_labels1"
    data = cycle.dataset(file_name, data_label)
    train(data, 50, date, epochs=1)
    data = pd.read_csv("./result/" + date + "result")
    baseline = data.loc[:, ["real", "imag", "block"]]
    modify = data.loc[:, ["fake_real", "fake_imag", "block"]]
    qam = data.loc[:, ["cons"]]
    label = data.loc[:, ["labels"]]
    #cls.qam_training(modify, qam, 50, 100, "test1_qam", date)
    #cls.symbol_training(modify, label, 50, 1000, "test1_symbol", date)
    #cls.qam_training(baseline, qam, 50, 100, "base_line_qam", date)
    #cls.symbol_training(baseline, qam, 50, 1000, "base_line_symbol", date)