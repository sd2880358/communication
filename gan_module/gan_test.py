import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import scipy.io as sc
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import time
from IPython.display import clear_output
import math

c_4 = [1,-1]
c_16 = [3,1,-1,-3]
c_16r = [-3,-1,1,3]
cons_4 = np.dot(np.sqrt(0.5),[[i,j]for i in c_4 for j in c_4])
cons_16 = [[i,j]for j in c_16 for i in c_16r]
LAMBDA = 10

def dataset(dataFile, labelFile):
    dataFile = "./communication/" + dataFile
    labelFile = "./communication/" + labelFile
    my_data = sc.loadmat(dataFile)
    my_labels = sc.loadmat(labelFile)
    my_data = my_data['Y']
    X = my_labels['L_S_x'].T
    cons = my_labels['L_Constellations'][0]
    data = table_data(my_data)
    label = assign_labels(X, cons)
    return data, label

def assign_labels(X, cons):
    for i in range(len(cons)):
        if cons[i] == 2:
            X[i] = X[i] + 4
    X = X-1
    return X

def table_data(my_data):
    data = np.dstack([my_data.real.T, my_data.imag.T])
    return data

data1 = "hard"
data1_label = "hard_label"
data, label = dataset(data1, data1_label)
test_data = data.reshape(1000,1,50,2)
test_label = label.reshape(1000,1,50,1)
test_label = tf.cast(test_label, tf.float32)

BUFFER_SIZE = 1000
BATCH_SIZE = 1
SIGNAL_SIZE = 2
BlockSize = 50
Lambda = 10

def random_crop(image):
    cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*2*1, use_bias=False, input_shape=[50,2]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((5, 5, 100)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(50))
    model.add(layers.Reshape((50,1)))
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Reshape((5, 5, 2)))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[5, 5, 2]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def cal_cycle_loss(real, fake):
    loss1 = tf.reduce_mean(tf.abs(real - fake))
    return LAMBDA * loss1

generator_optimizer = tf.keras.optimizers.Adam(1e-2)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-2)
cycle_optimizer = tf.keras.optimizers.Adam(1e-2)



EPOCHS = 50


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(noise, label):
    with tf.GradientTape(persistent=True) as tape:
        generated = generator(noise, training=True)
        real_output = discriminator(label, training=True)
        fake_output = discriminator(generated, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        cycle_loss = cal_cycle_loss(label, generated)
        total_loss = cycle_loss + gen_loss

    gradients_of_generator = tape.gradient(total_loss, generator.trainable_variables)
    gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(test_data, test_label, epochs):
    for epoch in range(epochs):
        start = time.time()
        fake_array = []
        for i in range(len(test_label)):
            noise = test_data[i]
            label = test_label[i]
            train_step(noise, label)
            fake_array.append(generator(noise).numpy())
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print(cal_cycle_loss(fake_array, test_label))
generator = make_generator_model()
discriminator = make_discriminator_model()
checkpoint_path = "./checkpoints/train"
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
mlp = train(test_data, test_label, EPOCHS)
