import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os,sys
import scipy.io as sc
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
import tensorflow_probability as tfp
import time
from IPython.display import clear_output
import math
def dataset(dataFile, labelFile):
    dataFile = "../ML_Symbol_Gen-main/" + dataFile
    labelFile = "../ML_Symbol_Gen-main/" + labelFile
    my_data = sc.loadmat(dataFile)
    my_labels = sc.loadmat(labelFile)
    my_data = my_data['Y']
    label = my_labels['L_S_x']
    label_real = my_labels['X'].real
    label_imag = my_labels['X'].imag
    noise = my_labels['N']
    interference = my_labels['L_Interference'][0]
    myOrig = table_data(my_data, my_labels['L_Constellations'][0], label, interference, noise,
                       label_real, label_imag)
    mytable = assign_label(myOrig)
    return mytable


def assign_label(data):
    myTest = data.copy()
    myTest.loc[myTest.cons == 2, 'label'] = myTest.loc[myTest.cons == 2, 'label'] + 4
    myTest.label = myTest.label - 1
    return myTest


def table_data(my_data, cons, label, interference, noise, label_real, label_imag):
    block = my_data.shape[1]
    my_data_size = my_data.shape[0] * block
    my_data_div = my_data.T.reshape(my_data_size, )
    label_real = label_real.T.reshape(my_data_size, )
    label_imag = label_imag.T.reshape(my_data_size, )
    noise = noise.T.reshape(my_data_size, )
    cons_array = np.array([[cons[i]] * my_data.shape[0] for i in range(0, block)]).reshape(my_data_size, )
    block_array = np.array([([i + 1] * my_data.shape[0]) for i in range(0, block)]).reshape(my_data_size, )
    interference_array = np.array([[interference[i]] * my_data.shape[0]
                               for i in range(0, block)]).reshape(my_data_size, )
    label_array = label.T.reshape(my_data_size, )
    test_pd = pd.DataFrame({'real': my_data_div.real, 'imag': my_data_div.imag,
                            'cons': cons_array, 'block': block_array,
                            'label': label_array,
                           'interference':interference_array,
                           'N_R': noise.real, 'N_I':noise.imag,
                           'label_real': label_real, 'label_imag':label_imag})
    return test_pd


def make_generator(blockSize):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (1, 2), strides=(1,2), activation="linear",
                            input_shape=(blockSize, 2, 1)))
    model.add(layers.Conv2D(32, (1,16), activation="linear", padding='same'))
    model.add(layers.Conv2D(16, (1,32), activation="linear", padding='same'))
    model.add(layers.Reshape((blockSize, 16, 1)))
    model.add(layers.AveragePooling2D((1,8)))
    model.add(layers.Dense(1))
    return model



def make_discriminator_model(blockSize):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (2, 1), strides=(1, 1), padding='same',
                                     input_shape=(blockSize, 2, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model  = model
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

def classifier(blockSize):
    input_shape = (blockSize, 2, 1)
    c_input = keras.layers.Input(shape=input_shape)
    cn_1 = layers.Conv2D(64, (2, 1), padding='same')(c_input)
    dr_1 = layers.Dropout(0.3)(cn_1)
    cn_2 = layers.Conv2D(128, (2, 1), padding='same')(dr_1)
    dr_2 = layers.Dropout(0.3)(cn_2)
    clf_out = keras.layers.Dense(1,  activation="softmax")(dr_2)
    mu = keras.layers.Dense(1)(dr_2)
    sigma = keras.layers.Dense(1, activation=lambda x: tf.math.exp(x))(dr_2)
    model = keras.models.Model(inputs = c_input, outputs = [clf_out, mu, sigma])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mean_abs_loss = tf.keras.losses.MeanAbsoluteError()
categories_loss = tf.keras.losses.CategoricalCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def noise_loss(noise_output):
    return mean_abs_loss(0, noise_output)


def cat_loss(c, c_hat):
    return categories_loss(c, c_hat)

def shuffle_data(my_table, blockSize):
    '''
    real_y = (2*my_table.real.min())/(my_table.real.max() - my_table.real.min()) + 1
    real_x = (my_table.real.max()) / (1 + real_y)
    imag_y = (2*my_table.imag.min())/(my_table.imag.max() - my_table.imag.min()) + 1
    imag_x = (my_table.imag.max()) / (1 + imag_y)
    my_table.real = (my_table.real / real_x) - real_y
    my_table.imag = (my_table.imag/ imag_x) - imag_y
    '''
    train_feature = my_table.loc[:, ('real', 'imag')]
    train_label = my_table.loc[:, ('label_real', 'label_imag')]
    noise = my_table.loc[:, ('N_R', 'N_I')]
    test_feature = tf.cast(train_feature, tf.float32)
    test_label = tf.cast(train_label, tf.float32)
    test_noise = tf.cast(noise, tf.float32)
    block = int(test_feature.shape[0]/blockSize)
    test_feature = tf.reshape(test_feature,(block,blockSize,2,1))
    test_label = tf.reshape(test_label, (block,blockSize,2,1))
    test_noise = tf.reshape(test_noise, (block,blockSize,2,1))
    symbol = my_table.loc[:, 'label']
    symbol = tf.reshape(symbol, (block, blockSize, 1))
    return test_feature, test_label, symbol, test_noise

def start_train(BATCH_SIZE, BUFFER_SIZE, data, filePath):
    @tf.function
    def train_step(total, label):
        g_noise = tf.random.normal([BATCH_SIZE, blockSize, 2, 1])
        with tf.GradientTape(persistent=True) as tape:
            s = generator_s(g_noise, training=True)
            fake_n = generator_n(g_noise, training=True)
            i = generator_i(g_noise, training=True)
            gen = (s + fake_n + i)
            c_hat, mu, sigma = classifier_t(gen)
            fake_t = discriminator_t(gen, training=True)
            real_t = discriminator_t(total, training=True)
            gen_loss = generator_loss(fake_t)
            fake_d = discriminator_d(s, training=True)
            real_d = discriminator_d(label, training=True)
            gen_s_loss = generator_loss(fake_d)
            disc_t_loss = discriminator_loss(real_t, fake_t)
            disc_d_loss = discriminator_loss(real_d, fake_d)
            class_t_loss = cat_loss(c_hat, s)
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            c_1_loss = tf.reduce_mean(-dist.log_prob(g_noise))
            total_gen_loss = gen_loss + (class_t_loss + 0.1 * c_1_loss)
            total_s_loss = gen_s_loss
            total_n_loss = total_gen_loss
            total_i_loss = total_gen_loss
            q_loss = class_t_loss +  0.1 * c_1_loss
        gradients_of_s_generator = tape.gradient(total_s_loss, generator_s.trainable_variables)
        gradients_of_i_generator = tape.gradient(total_i_loss, generator_i.trainable_variables)
        gradients_of_n_generator = tape.gradient(total_n_loss, generator_n.trainable_variables)
        gradients_of_classifier_t = tape.gradient(q_loss, classifier_t.trainable_variables)
        gradients_of_discriminator_t = tape.gradient(disc_t_loss, discriminator_t.trainable_variables)
        gradients_of_discriminator_d = tape.gradient(disc_d_loss, discriminator_d.trainable_variables)
        generator_s_optimizer.apply_gradients(zip(gradients_of_s_generator, generator_s.trainable_variables))
        generator_i_optimizer.apply_gradients(zip(gradients_of_i_generator, generator_i.trainable_variables))
        generator_n_optimizer.apply_gradients(zip(gradients_of_n_generator, generator_n.trainable_variables))
        classifier_t_optimizer.apply_gradients(zip(gradients_of_classifier_t, classifier_t.trainable_variables))
        discriminator_t_optimizer.apply_gradients(zip(gradients_of_discriminator_t, discriminator_t.trainable_variables))
        discriminator_d_optimizer.apply_gradients(zip(gradients_of_discriminator_d, discriminator_d.trainable_variables))
    checkpoint_path = "./checkpoints/test4/01_17/" + filePath
    ckpt = tf.train.Checkpoint(generator_s=generator_s,
                               generator_n=generator_n,
                               generator_i=generator_i,
                               classifier_t = classifier_t,
                               discriminator_t=discriminator_t,
                               discriminator_d=discriminator_d,
                               generator_s_optimizer=generator_s_optimizer,
                               generator_n_optimizer=generator_n_optimizer,
                               generator_i_optimizer=generator_i_optimizer,
                               classifier_t_optimizer=classifier_t_optimizer,
                               discriminator_d_optimizer=discriminator_d_optimizer,
                               discriminator_t_optimizer=discriminator_t_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    feature, labels, symbol, noise = shuffle_data(data, BUFFER_SIZE)
    train_f = tf.data.Dataset.from_tensor_slices(feature).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_l = tf.data.Dataset.from_tensor_slices(labels).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_n = tf.data.Dataset.from_tensor_slices(noise).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    for epoch in range(EPOCHS):
        start = time.time()
        n = 0
        for i, j in tf.data.Dataset.zip((train_f, train_l)):
            train_step(i, j)
        if n % 10 == 0:
            print('.', end='')
            n += 1
        if epoch == EPOCHS-1:
            fake_c, mu, sigma = classifier_t(feature)
            sample = tf.random.normal([1000, blockSize, 2, 1])
            fake_s = generator_s(feature)
            fake_i = generator_i(feature)
            fake_n = generator_n(feature)
            print(fake_s)
            fake_mixed = fake_s + fake_i + fake_n
            fake_t = discriminator_t(fake_mixed)
            fake_d = discriminator_d(fake_s)
            real_t = discriminator_t(feature)
            real_d = discriminator_d(labels)
            mixed_loss = generator_loss(fake_mixed)
            fake_s_loss = generator_loss(fake_s)
            categorical_c_loss = cat_loss(fake_c, labels)
            print("_____Test Result:_____")
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
            print('The generator total loss is', mixed_loss)
            print('The signal loss is ', fake_s_loss)
            print("the signal categories loss is", categorical_c_loss)
            print("___________________\n")
            data = pd.DataFrame({
                "mixed loss": mixed_loss,
                "signal loss": fake_s_loss,
                "categories loss": categorical_c_loss
            }, index=[0])
            data.to_csv("./result/test_1_16/"+filePath)



if __name__ == '__main__':
    EPOCHS = 100
    generator_s_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_n_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_i_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_t_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    classifier_t_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for i in range(5,6):
        blockSize = i*10
        i = str(i)
        data = "my_data" + i
        data_label = "my_labels" + i
        file_directory = 'method' + i
        generator_s = make_generator(blockSize)
        generator_n = make_generator(blockSize)
        generator_i = make_generator(blockSize)
        discriminator_t = make_discriminator_model(blockSize)
        discriminator_d = make_discriminator_model(blockSize)
        classifier_t = classifier(blockSize)
        data_table = dataset(data, data_label)
        start_train(250, blockSize, data_table, file_directory)