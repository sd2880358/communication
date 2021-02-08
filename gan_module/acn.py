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
    model.add(layers.Conv2D(32, (5, 2), strides=(2, 2), activation="linear", padding='same',
                            input_shape=(blockSize, 2, 1)))
    model.add(layers.Conv2D(16, (5, 1), activation="linear", padding='same'))
    model.add(layers.Conv2D(8, kernel_size=(5, 1), activation='linear', padding='same'))
    model.add(layers.Conv2DTranspose(8, kernel_size=(5, 1), activation='linear', padding='same'))
    model.add(layers.Conv2DTranspose(16, kernel_size=(5, 2), strides=(2, 2), activation='linear', padding='same'))
    model.add(layers.Conv2DTranspose(32, kernel_size=(5, 2), activation='linear', padding='same'))
    model.add(layers.Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same'))
    return model

def make_classifier(blocksize):
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (1, 2), padding='same', activation='relu', input_shape=(blocksize, 2, 1)))
    model.add(layers.MaxPooling2D((1, 1)))
    model.add(layers.Conv2D(32, (4,1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((1, 1)))
    model.add(layers.Conv2D(64, (4,1), strides=(1,2),padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((1, 1)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(20))
    return model




sparse_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
mean_abs_loss = tf.keras.losses.MeanAbsoluteError()


def classifier_loss(fake, label):
    fake_loss = sparse_entropy(label, fake)
    total_loss = fake_loss
    return total_loss


def generator_loss(fake_output, label):
    return sparse_entropy(label, fake_output)

def identity_loss(real, fake):
    loss =  tf.losses.mean_absolute_error(real, fake)
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

def get_total_loss(predict_label, true_label):
    sum = 0
    for i in range(predict_label.shape[0]):
        for j in range(predict_label.shape[1]):
            prediction = np.argmax(predict_label[i,j])
            if prediction != true_label[i,j]:
                sum += 1
    return 1 - sum/(predict_label.shape[0] * predict_label.shape[1])


def start_train(BATCH_SIZE, BUFFER_SIZE, data, filePath):
    @tf.function
    def train_step(total, label, symbol):
        with tf.GradientTape(persistent=True) as tape:
            s = generator(total, training=True)
            fake_s = classifier(s, training=True)
            id_loss = identity_loss(s, label)
            cls_loss = classifier_loss(fake_s, symbol)
            gen_total_loss = id_loss + cls_loss
        gradients_of_generator = tape.gradient(gen_total_loss, generator.trainable_variables)
        gradients_of_classifier = tape.gradient(cls_loss, classifier.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        classifier_optimizer.apply_gradients(
            zip(gradients_of_classifier, classifier.trainable_variables))

    checkpoint_path = "./checkpoints/test5/" + date + filePath
    ckpt = tf.train.Checkpoint(generator = generator,
                               classifier = classifier,
                               generator_optimizer = generator_optimizer,
                               classifier_optimizer = classifier_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    feature, labels, symbol, my_table = shuffle_data(data, BUFFER_SIZE)
    train_f = tf.data.Dataset.from_tensor_slices(feature).batch(BATCH_SIZE)
    train_l = tf.data.Dataset.from_tensor_slices(labels).batch(BATCH_SIZE)
    train_s = tf.data.Dataset.from_tensor_slices(symbol).batch(BATCH_SIZE)
    disen_hist = []
    for epoch in range(EPOCHS):
        start = time.time()
        n = 0
        for i, j, k in tf.data.Dataset.zip((train_f, train_l, train_s)):
            train_step(i, j, k)
        if n % 10 == 0:
            print('.', end='')
            n += 1

        if (epoch + 1) % 10 == 0:
            fake_s = generator(feature)
            id_loss = abs(fake_s - labels).numpy().mean()
            relative_loss = np.median(abs((labels - fake_s) / labels))
            disen_Loss = [id_loss,relative_loss]
            disen_hist.append(disen_Loss)

        if (epoch + 1) % 100 == 0 :
            fake_s = generator(feature)
            ## measuring the absolute loss between generator and disentanglement
            print("result of signal and interference")
            print("---")
            print("result of fake signal", fake_s[1,1,1])
            print("actual feature", feature[1,1,1])
            print("actual labels", labels[1,1,1])
            disen_hist.reverse()
            test_hist = np.array(disen_hist)
            result = pd.DataFrame(
                  {
                "real":feature.numpy()[:,:,0].flatten(),
                "imag": feature.numpy()[:,:,1].flatten(),
                "fake_real": fake_s.numpy()[:, :, 0].flatten(),
                "fake_imag": fake_s.numpy()[:,:, 1].flatten(),
                "block":data_table.block,
                "cons": (my_table.cons.to_numpy()-1).flatten(),
                "label_real":my_table.label_real,
                "label_imag":my_table.label_imag,
                "labels": (my_table.label.to_numpy()).flatten()}
            )
            # relative loss between fake signal and signal_hat
            '''
            id_loss = abs(fake_s - fake_c).numpy().mean()
            relative_loss = np.median(abs((fake_s - fake_c) / fake_c))
            '''
            id_loss = abs(fake_s - labels).numpy().mean()
            prediction = classifier.predict(fake_s)
            relative_loss = generator_loss(prediction, symbol)
            test_acc = get_total_loss(prediction, symbol)
            print("_____Test Result:_____")
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
            print('The disentangle total loss is', id_loss)
            print('The relative loss is ', relative_loss)
            print('The classifier accuracy is', test_acc)
            print("___________________\n")
            data = pd.DataFrame({
                "disentangle loss": test_hist[:,0],
                "relative loss": test_hist[:, 1]
            })
            result.to_csv("./result/" + date + filePath+"result", index=False)
            data.to_csv("./result/" + date + filePath)


if __name__ == '__main__':
    EPOCHS = 500
    LAMBDA = 10
    date = "2_7/"
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    classifier_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for i in range(1, 2):
        blockSize = 50
        i = str(i)
        data = "my_data" + i
        data_label = "my_labels" + i
        file_directory = 'acn'
        generator = make_generator(blockSize)
        classifier = make_classifier(blockSize)
        data_table = dataset(data, data_label)
        start_train(250, blockSize, data_table, file_directory)
        data = pd.read_csv("./result/"+date+file_directory+"result")
        baseline = data.loc[:, ["real", "imag", "block"]]
        modify = data.loc[:, ["fake_real", "fake_imag", "block"]]
        qam = data.loc[:, ["cons"]]
        label = data.loc[:, ["labels"]]
        #cls.qam_training(modify, qam, 50, 100, "test1_qam")
        #cls.symbol_training(baseline, label, 50, 1000, "test1_symbol")
