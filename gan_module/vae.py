import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import random
import time
import cycleGAN as cycle

class Reparameterize(layers.Layer):
    def call(self, inputs):
        Z_mu, Z_logvar = inputs
        epsilon = tf.random.normal(tf.shape(Z_mu))
        sigma = tf.math.exp(0.5*Z_logvar)
        return Z_mu + sigma * epsilon

class CVAE(Model):
    def __init__(self, input_shape, latent_dim=50, learning_rate=1e-4):
        super(CVAE, self).__init__()
        self.C = 0
        self.latent_dim = latent_dim
        self.gamma = 100
        encoder_input = layers.Input(shape=input_shape)
        X = layers.Conv2D(4, (5, 2), strides=(2, 2), activation="relu", padding='same')(encoder_input)
        X = layers.Conv2D(2, (5, 1), activation="relu", padding='same')(X)
        X = layers.Flatten()(X)
        Z_mu = layers.Dense(self.latent_dim)(X)
        Z_logvar = layers.Dense(self.latent_dim, activation='relu')(X)
        Z = Reparameterize()([Z_mu, Z_logvar])


        decode_input = layers.Input(shape=latent_dim)
        X = layers.Reshape((25,1,2))(decode_input)
        X = layers.Conv2DTranspose(2, kernel_size=(5, 1), activation='relu', padding='same')(X)
        X = layers.Conv2DTranspose(4, kernel_size=(5, 2), strides=(2, 2), activation='relu', padding='same')(X)
        decode_output = layers.Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same')(X)
        self.encoder = Model(encoder_input, [Z_mu, Z_logvar, Z])
        self.decoder = Model(decode_input, decode_output)
        self.vae = Model(encoder_input, self.decoder(Z))
        self.vae.compile(optimizer='adam', loss=loss, metrics=[reconstruction_loss, kl_divergence])
    def predict(self, inputs):
        return self.vae.predict(inputs)


def reconstruction_loss(X, X_pred, input_shape):
    mse = tf.losses.MeanSquaredError()
    return mse(X, X_pred) * np.prod(input_shape)

def kl_divergence(Z_logvar, Z_mu, C, gamma):
    C += (1/1440)
    C = min(C, 35)
    kl = -0.5 * tf.reduce_mean(1 + Z_logvar - Z_mu**2 - tf.math.exp(Z_logvar))
    return gamma * tf.math.abs(kl - C)

def loss(X, X_pred):
   return reconstruction_loss(X, X_pred) + kl_divergence(X, X_pred)





def start_train(BATCH_SIZE, BUFFER_SIZE, data, input_shape, filePath):
    C = 0
    gamma = 100

    @tf.function
    def train_step(total, label):
        with tf.GradientTape(persistent=True) as tape:
            Z_mu, Z_logvar, Z = model.encoder(total)
            X_pred = model.decoder(Z)
            reg_loss = reconstruction_loss(label, X_pred, input_shape)
            kl_loss = kl_divergence(Z_logvar, Z_mu, C, gamma)
            total_loss = reg_loss + kl_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/" + date + filePath
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    feature, labels, symbol, my_table = cycle.shuffle_data(data, BUFFER_SIZE)
    train_f = tf.data.Dataset.from_tensor_slices(feature).batch(BATCH_SIZE)
    train_l = tf.data.Dataset.from_tensor_slices(labels).batch(BATCH_SIZE)
    for epoch in range(epochs):
        start = time.time()
        n = 0
        for i, j in tf.data.Dataset.zip((train_f, train_f)):
            train_step(i, j)
        if n % 10 == 0:
            print('.', end='')
            n += 1
        if epoch % 10 == 0:
            predicted = model.vae.predict(feature)
            error = reconstruction_loss(feature, predicted, input_shape)
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
            print(predicted)







if __name__ == '__main__':
    date = "2_1/"
    file_name = "my_data1"
    data_label = "my_labels1"
    file_path = "beta_vae"
    data = cycle.dataset(file_name, data_label)
    model = CVAE(input_shape=(50,2,1), latent_dim=25)
    encoder = model.encoder
    decoder = model.decoder
    epochs = 100
    input_shape = (50, 2, 1)
    batchSize = 250
    optimizer = tf.keras.optimizers.Adam(1e-4)
    start_train(250, 50, data, input_shape, file_path)
