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
    model.add(layers.Dense(100, input_shape=(blockSize * 2+1,)))
    model.add(layers.Reshape([blockSize, 2, 1]))
    model.add(layers.Conv2D(16, (1, 2), strides=(1,2), activation="linear"))
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
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

def classifier(blockSize, latent_dim=50):
    input_shape = (blockSize, 2, 1)
    c_input = keras.layers.Input(shape=input_shape)
    cn_1 = layers.Conv2D(64, (2, 1), padding='same')(c_input)
    dr_1 = layers.Dropout(0.3)(cn_1)
    cn_2 = layers.Conv2D(128, (2, 1), padding='same')(dr_1)
    dr_2 = layers.Dropout(0.3)(cn_2)
    flatten = layers.Flatten()(dr_2)
    clf_out = keras.layers.Dense(latent_dim, activation="softmax")(flatten)
    mu = keras.layers.Dense(1)(flatten)
    sigma = keras.layers.Dense(1, activation=lambda x: tf.math.exp(x))(flatten)
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


def create_gen_input(batch_size, noise_size, n_class, seed=None):
        # create noise input
        noise = tf.random.normal([batch_size, noise_size], seed=seed)
        # Create categorical latent code
        label = tf.random.uniform([batch_size], minval=0, maxval=10, dtype=tf.int32, seed=seed)
        label = tf.one_hot(label, depth=n_class)
        # Create one continuous latent code
        c_1 = tf.random.uniform([batch_size, 1], minval=-1, maxval=1, seed=seed)
        #latent code, continuous latent code, noise
        return label, c_1, noise

def concat_inputs(input):
    concat_input = keras.layers.Concatenate()(input)
    return concat_input

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

def start_train(BATCH_SIZE, BUFFER_SIZE, data, filePath, date):
    @tf.function
    def train_step(total, label):
        c_label, c_1, g_noise = create_gen_input(BATCH_SIZE, 50, 50)
        train_gen = concat_inputs([c_label, c_1, g_noise])
        with tf.GradientTape(persistent=True) as tape:
            fake_g = generator(train_gen, training=True)
            c_hat, mu, sigma = classifier(fake_g, training=True)
            fake_t = discriminator(fake_g, training=True)
            real_t = discriminator(label, training=True)
            gen_loss = generator_loss(fake_t)
            disc_loss = discriminator_loss(real_t, fake_t)
            #Auxiliary loss
            class_loss = cat_loss(c_label, c_hat)
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            c_1_loss = tf.reduce_mean(-dist.log_prob(c_1))
            total_gen_loss = gen_loss + (class_loss + 0.1 * c_1_loss)
            q_loss = class_loss +  0.1 * c_1_loss
        gradients_of_generator = tape.gradient(total_gen_loss, generator.trainable_variables)
        gradients_of_classifier = tape.gradient(q_loss, classifier.trainable_variables)
        gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        classifier_optimizer.apply_gradients(zip(gradients_of_classifier, classifier.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    checkpoint_path = "./checkpoints/test4/"+ date + filePath
    ckpt = tf.train.Checkpoint(generator=generator,
                               classifier = classifier,
                               discriminator=discriminator,
                               generator_optimizer=generator_optimizer,
                               classifier_optimizer=classifier_optimizer,
                               discriminator_optimizer=discriminator_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    feature, labels, symbol, noise = shuffle_data(data, BUFFER_SIZE)
    train_f = tf.data.Dataset.from_tensor_slices(feature).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_l = tf.data.Dataset.from_tensor_slices(labels).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    for epoch in range(EPOCHS):
        start = time.time()
        n = 0
        for i, j in tf.data.Dataset.zip((train_f, train_l)):
            train_step(i, j)
        if n % 10 == 0:
            print('.', end='')
            n += 1
        if epoch == EPOCHS-1:
            g_label, c_1, noise = create_gen_input(10000, 50, 50)
            test_input = concat_inputs([g_label, c_1, noise])
            fake_signal = generator(test_input)
            fake_s = discriminator(fake_signal)
            real_s = discriminator(labels)
            mixed_loss = generator_loss(fake_s).numpy()
            dis_loss = discriminator_loss(real_s, fake_s).numpy()
            print("_____Test Result:_____")
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
            print('The generator total loss is', mixed_loss)
            print('The discriminator loss is ', dis_loss)
            print("___________________\n")
            result = pd.DataFrame(
                  {
                "fake_real": fake_signal.numpy()[:, :, 0].flatten(),
                "fake_imag": fake_signal.numpy()[:,:, 1].flatten(),
                "block":data_table.block,
                "label_real":labels.numpy()[:, :, 0].flatten(),
                "label_imag":labels.numpy()[:, :, 1].flatten()}
            )
            data = pd.DataFrame({
                "gen loss": mixed_loss,
                'dis_loss': dis_loss
            }, index=[0])
            print(fake_signal)
            data.to_csv("./result/"+ date+ filePath)
            result.to_csv("./result/" + date + filePath + "info")



if __name__ == '__main__':
    EPOCHS = 1
    date = "2_9/"
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    classifier_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for i in range(1,2):
        blockSize = 50
        i = str(i)
        data = "my_data" + i
        data_label = "my_labels" + i
        file_directory = 'info' + i
        generator = make_generator(blockSize)
        discriminator = make_discriminator_model(blockSize)
        classifier = classifier(blockSize)
        data_table = dataset(data, data_label)
        start_train(250, blockSize, data_table, file_directory, date)