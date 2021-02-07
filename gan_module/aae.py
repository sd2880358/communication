import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import scipy.io as sc
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
import tensorflow_probability as tfp
import time
from IPython.display import clear_output
import math
import cnn_classifier as cls


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
                            'interference': interference_array,
                            'N_R': noise.real, 'N_I': noise.imag,
                            'label_real': label_real, 'label_imag': label_imag})
    return test_pd


def make_generator(blockSize):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (5, 2), strides=(2, 2), activation="linear", padding='same',
                            input_shape=(blockSize, 2, 1)))
    model.add(layers.Conv2D(8, (5, 1), activation="linear", padding='same'))
    model.add(layers.Conv2DTranspose(8, kernel_size=(5, 1), activation='linear', padding='same'))
    model.add(layers.Conv2DTranspose(16, kernel_size=(5, 2), strides=(2, 2), activation='linear', padding='same'))
    model.add(layers.Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same'))
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




cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mean_abs_loss = tf.keras.losses.MeanSquaredError()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)




def identity_loss(real, fake):
    loss =  mean_abs_loss(real , fake)
    return LAMBDA * 0.5 * loss


def shuffle_data(my_table, blockSize):
    '''
    real_y = (2*my_table.real.min())/(my_table.real.max() - my_table.real.min()) + 1
    real_x = (my_table.real.max()) / (1 + real_y)
    imag_y = (2*my_table.imag.min())/(my_table.imag.max() - my_table.imag.min()) + 1
    imag_x = (my_table.imag.max()) / (1 + imag_y)
    my_table.real = (my_table.real / real_x) - real_y
    my_table.imag = (my_table.imag/ imag_x) - imag_y
    '''
    groups = [df for _, df in my_table.groupby('block')]
    random.shuffle(groups)
    my_table = pd.concat(groups).reset_index()
    train_feature = my_table.loc[:, ('real', 'imag')]
    train_label = my_table.loc[:, ('label_real', 'label_imag')]
    noise = my_table.loc[:, ('N_R', 'N_I')]
    test_feature = tf.cast(train_feature, tf.float32)
    test_label = tf.cast(train_label, tf.float32)
    test_noise = tf.cast(noise, tf.float32)
    block = int(test_feature.shape[0] / blockSize)
    test_feature = tf.reshape(test_feature, (block, blockSize, 2, 1))
    test_label = tf.reshape(test_label, (block, blockSize, 2, 1))
    test_noise = tf.reshape(test_noise, (block, blockSize, 2, 1))
    symbol = my_table.loc[:, 'label']
    symbol = tf.reshape(symbol, (block, blockSize, 1))
    return test_feature, test_label, symbol, my_table


def start_train(BATCH_SIZE, BUFFER_SIZE, data, filePath):
    @tf.function
    def train_step(total, label):
        with tf.GradientTape(persistent=True) as tape:
            s = generator_s(total, training=True)
            fake_u = generator_u(label, training=True)
            gen = (s + fake_u)
            u_k = total - label
            result_fake_u = discriminator_u(fake_u, training=True)
            result_real_u = discriminator_u(u_k, training=True)
            fake_t = discriminator_t(gen, training=True)
            real_t = discriminator_t(total, training=True)
            gen_loss = generator_loss(fake_t)
            fake_d = discriminator_d(s, training=True)
            real_d = discriminator_d(label, training=True)
            gen_s_loss = generator_loss(fake_d)
            gen_u_loss = generator_loss(result_fake_u)
            disc_t_loss = discriminator_loss(real_t, fake_t)
            disc_d_loss = discriminator_loss(real_d, fake_d)
            disc_u_loss = discriminator_loss(result_real_u, result_fake_u)
            id_s_loss = identity_loss(s , label)
            id_u_loss = identity_loss(fake_u, u_k)
            total_s_loss = gen_s_loss * 0.5 +  gen_loss + id_s_loss
            total_u_loss = gen_u_loss * 0.5 + gen_loss + id_u_loss
        gradients_of_s_generator = tape.gradient(total_s_loss, generator_s.trainable_variables)
        gradients_of_u_generator = tape.gradient(total_u_loss, generator_u.trainable_variables)
        gradients_of_discriminator_t = tape.gradient(disc_t_loss, discriminator_t.trainable_variables)
        gradients_of_discriminator_d = tape.gradient(disc_d_loss, discriminator_d.trainable_variables)
        gradients_of_discriminator_u = tape.gradient(disc_u_loss, discriminator_u.trainable_variables)
        generator_s_optimizer.apply_gradients(zip(gradients_of_s_generator, generator_s.trainable_variables))
        generator_u_optimizer.apply_gradients(zip(gradients_of_u_generator, generator_u.trainable_variables))
        discriminator_u_optimizer.apply_gradients(
            zip(gradients_of_discriminator_u, discriminator_u.trainable_variables))
        discriminator_t_optimizer.apply_gradients(
            zip(gradients_of_discriminator_t, discriminator_t.trainable_variables))
        discriminator_d_optimizer.apply_gradients(
            zip(gradients_of_discriminator_d, discriminator_d.trainable_variables))

    checkpoint_path = "./checkpoints/test5/" + date + filePath
    ckpt = tf.train.Checkpoint(generator_s=generator_s,
                               generator_u=generator_u,
                               discriminator_t=discriminator_t,
                               discriminator_d=discriminator_d,
                               discriminator_u = discriminator_u,
                               generator_s_optimizer=generator_s_optimizer,
                               generator_u_optimizer=generator_u_optimizer,
                               discriminator_u_optimizer = discriminator_u_optimizer,
                               discriminator_d_optimizer=discriminator_d_optimizer,
                               discriminator_t_optimizer=discriminator_t_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    feature, labels, symbol, my_table = shuffle_data(data, BUFFER_SIZE)
    train_f = tf.data.Dataset.from_tensor_slices(feature).batch(BATCH_SIZE)
    train_l = tf.data.Dataset.from_tensor_slices(labels).batch(BATCH_SIZE)
    disen_hist = []
    for epoch in range(EPOCHS):
        start = time.time()
        n = 0
        for i, j in tf.data.Dataset.zip((train_f, train_l)):
            train_step(i, j)
        if n % 10 == 0:
            print('.', end='')
            n += 1

        if epoch % 10 == 0:
            fake_s = generator_s(feature)
            id_loss = abs(fake_s - labels).numpy().mean()
            relative_loss = np.median(abs((labels - fake_s) / labels))
            disen_Loss = [id_loss,relative_loss]
            disen_hist.append(disen_Loss)

        if epoch == EPOCHS - 1:

            ## measuring the absolute loss between generator and disentanglement

            fake_c = generator_s(feature)
            fake_u = generator_u(feature)
            print("result of signal and interference")
            print("---")
            print("result of fake signal", fake_c[1,1,1])
            print("result of fake unknown", fake_u[1,1,1])
            fake_mixed = fake_c + fake_u
            print("result of fake mixed", fake_mixed[1,1,1])
            print("actual feature", feature[1,1,1])
            print("actual labels", labels[1,1,1])
            disen_hist.reverse()
            test_hist = np.array(disen_hist)
            result = pd.DataFrame(
                  {
                "real":feature.numpy()[:,:,0].flatten(),
                "imag": feature.numpy()[:,:,1].flatten(),
                "fake_real": fake_c.numpy()[:, :, 0].flatten(),
                "fake_imag": fake_c.numpy()[:,:, 1].flatten(),
                "block":data_table.block,
                "cons": (my_table.cons.to_numpy()-1).flatten(),
                "labels_real": labels.numpy()[:,:,0].flatten(),
                "labels_imag": labels.numpy()[:,:,1].flatten(),
                "labels": (my_table.label.to_numpy()).flatten()}
            )
            # relative loss between fake signal and signal_hat
            '''
            id_loss = abs(fake_s - fake_c).numpy().mean()
            relative_loss = np.median(abs((fake_s - fake_c) / fake_c))
            '''
            id_loss = abs(fake_c - labels).numpy().mean()
            relative_loss = tf.losses.MeanAbsolutePercentageError()(fake_c, labels)
            '''
            sample = tf.random.normal([1000, blockSize, 2, 1])
            fake_s = generator_s(sample)
            fake_i = generator_i(sample)
            fake_n = generator_n(sample)
            fake_mixed = fake_s + fake_i + fake_n
            print(fake_mixed[:,1,1])
            fake_t = discriminator_t(fake_mixed)
            fake_d = discriminator_d(fake_s)
            gen_total_loss = generator_loss(fake_t)
            gen_s_loss = generator_loss(fake_d)
            print(gen_total_loss)
            print(gen_s_loss)
            '''
            print("_____Test Result:_____")
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
            print('The disentangle total loss is', id_loss)
            print('The relative loss is ', relative_loss)
            print("___________________\n")
            data = pd.DataFrame({
                "disentangle loss": test_hist[:,0],
                "relative loss": test_hist[:, 1]
            })
            result.to_csv("./result/" + date + filePath+"result", index=False)
            data.to_csv("./result/" + date + filePath)


if __name__ == '__main__':
    EPOCHS = 1
    LAMBDA = 10
    date = "2_7/"
    generator_s_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_u_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_t_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_u_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disentangle_t_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for i in range(1, 2):
        blockSize = 50
        i = str(i)
        data = "my_data" + i
        data_label = "my_labels" + i
        file_directory = 'method1'
        generator_s = make_generator(blockSize)
        generator_u = make_generator(blockSize)
        discriminator_t = make_discriminator_model(blockSize)
        discriminator_d = make_discriminator_model(blockSize)
        discriminator_u = make_discriminator_model(blockSize)
        data_table = dataset(data, data_label)
        start_train(250, blockSize, data_table, file_directory)
        data = pd.read_csv("./result/"+date+file_directory+"result")
        baseline = data.loc[:, ["real", "imag", "block"]]
        modify = data.loc[:, ["fake_real", "fake_imag", "block"]]
        qam = data.loc[:, ["cons"]]
        label = data.loc[:, ["labels"]]
        #cls.qam_training(modify, qam, 50, 100, "test1_qam")
        #cls.symbol_training(modify, label, 50, 1000, "test1_symbol")
