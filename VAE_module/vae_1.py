import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os,sys
import scipy.io as sc
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import time
from IPython.display import clear_output
import math
def dataset(dataFile, labelFile):
    dataFile = "../ML_Symbol_Gen-main/" + dataFile
    labelFile = "../ML_Symbol_Gen-main/" + labelFile
    my_data = sc.loadmat(dataFile)
    my_labels = sc.loadmat(labelFile)
    my_data = my_data['Y']
    X = my_labels['L_S_x']
    myOrig = table_data(my_data, my_labels['L_Constellations'][0], X)
    mytable = assign_label(myOrig)
    return mytable


def assign_label(data):
    c_4 = [1,-1]
    c_16 = [3,1,-1,-3]
    c_16r = [-3,-1,1,3]
    cons_4 = np.dot(np.sqrt(0.5),[complex(i,j)for i in c_4 for j in c_4])
    cons_16 = np.array([complex(i,j)for j in c_16 for i in c_16r])
    cons_16 = cons_16/np.sqrt(np.mean(np.abs(cons_16)**2))
    cons4 = data[data.cons==1]
    cons4_label = np.array([[cons_4[i-1]]for i in cons4.label])
    cons16 = data[data.cons==2]
    cons16_label = np.array([[cons_16[i-1]]for i in cons16.label.to_numpy().real.astype(int)])
    data[data.cons==2].index
    data['buffer'] = 0
    data['buffer'] = 0
    data.iloc[data[data.cons==1].index, 5] = cons4_label
    data.iloc[data[data.cons==2].index, 5] = cons16_label
    data['label_real'] = data.buffer.to_numpy().real
    data['label_imag'] = data.buffer.to_numpy().imag
    myTest = data.copy()
    myTest.loc[myTest.cons == 2, 'label'] = myTest.loc[myTest.cons == 2, 'label'] + 4
    myTest.label = myTest.label - 1
    return myTest


def table_data(my_data, cons, label):
    block = my_data.shape[1]
    my_data_size = my_data.shape[0] * block
    my_data_div = my_data.T.reshape(my_data_size, )
    cons_array = np.array([[cons[i]] * my_data.shape[0] for i in range(0, block)]).reshape(my_data_size, )
    block_array = np.array([([i + 1] * my_data.shape[0]) for i in range(0, block)]).reshape(my_data_size, )
    label_array = label.T.reshape(my_data_size, )
    test_pd = pd.DataFrame({'real': my_data_div.real, 'imag': my_data_div.imag,
                            'cons': cons_array, 'block': block_array,
                            'label': label_array})
    return test_pd

def make_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, use_bias=False, input_shape=[50,2]))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def identity_loss(real, fake):
    loss = tf.reduce_mean(tf.abs(real - fake))
    return LAMBDA * 0.5 * loss


@tf.function
def train_step(total, label):
    with tf.GradientTape(persistent=True) as tape:
        s = generator_s(total, training=True)
        n = generator_n(total, training=True)
        i = generator_i(total, training=True)
        gen = (s + n + i)
        gen = tf.reshape(gen, (1,50,2))
        identity_s_loss = identity_loss(label, s)
        identity_g_loss = identity_loss(total, gen)
        identity_total_loss = identity_g_loss + identity_s_loss
        total_s_loss = identity_total_loss + 1/2 * gen_loss  + identity_s_loss
        total_n_loss = identity_total_loss + gen_loss
        total_i_loss = identity_total_loss + gen_loss

    gradients_of_s_generator = tape.gradient(total_s_loss, generator_s.trainable_variables)
    gradients_of_i_generator = tape.gradient(total_i_loss, generator_i.trainable_variables)
    gradients_of_n_generator = tape.gradient(total_n_loss, generator_n.trainable_variables)
    generator_s_optimizer.apply_gradients(zip(gradients_of_s_generator, generator_s.trainable_variables))
    generator_i_optimizer.apply_gradients(zip(gradients_of_i_generator, generator_i.trainable_variables))
    generator_n_optimizer.apply_gradients(zip(gradients_of_n_generator, generator_n.trainable_variables))

def shuffle_data(my_table):
    real_y = (2*my_table.real.min())/(my_table.real.max() - my_table.real.min()) + 1
    real_x = (my_table.real.max()) / (1 + real_y)
    imag_y = (2*my_table.imag.min())/(my_table.imag.max() - my_table.imag.min()) + 1
    imag_x = (my_table.imag.max()) / (1 + imag_y)
    my_table.real = (my_table.real / real_x) - real_y
    my_table.imag = (my_table.imag/ imag_x) - imag_y
    train_feature = data.loc[:, ('real', 'imag')]
    train_label = data.loc[:, ('label_real', 'label_imag')]
    test_feature = tf.cast(train_feature, tf.float32)
    test_label = tf.cast(train_label, tf.float32)
    test_feature = tf.reshape(test_feature,(1000,1,50,2))
    test_label = tf.reshape(test_label, (1000,1,50,2))
    symbol = data.loc[:, 'label']
    return test_feature, test_label

generator_s = make_generator()
generator_n = make_generator()
generator_i = make_generator()


generator_s_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_n_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_i_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_path = "./checkpoints/method1"

ckpt = tf.train.Checkpoint(generator_s=generator_s,
                           generator_n=generator_n,
                           generator_i=generator_i,
                           generator_s_optimizer=generator_s_optimizer,
                           generator_n_optimizer=generator_n_optimizer,
                           generator_i_optimizer=generator_i_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

LAMBDA = 10
EPOCHS = 500
data1 = "my_data"
data1_label = "my_labels"
data = dataset(data1, data1_label)
file_directory = './result/tes2/'
f, l = shuffle_data(data)

for epoch in range(EPOCHS):
    start = time.time()
    n = 0
    for i in range(len(f)):
        test = f[i]
        label = l[i]
        train_step(test, label)
        if n % 10 == 0:
            print('.', end='')
            n += 1

    if  (epoch+1)%5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print("_____Test Result:_____")
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
        id = str(epoch)
        s = generator_s(f, training=False)
        i = generator_i(s, training=False)
        n = generator_n(f, training=False)
        gen = s + i + n
        test = identity_loss(s, l)
        gen_loss = identity_loss(gen, f)

        print('The generator total loss is', gen_loss)
        print('The signal loss is ', test)
        print("___________________\n")