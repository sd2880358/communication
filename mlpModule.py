import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

dataFile = "./communication/my_data.mat"
sambolsFile = './communication/mysambol.mat'
labelFile = './communication/my_labels.mat'
my_data = sc.loadmat(dataFile)
my_labels = sc.loadmat(labelFile)
data2 = sc.loadmat(sambolsFile)
my_data = my_data['Y']
I = data2['I']
N = data2['N']
X = data2['X']

def table_data(my_data, snr, inr, cons, label):
    block = my_data.shape[0]
    my_data_size = my_data.shape[0] * block
    my_data_div = my_data.T.reshape(my_data_size,)
    snr_array = np.array([snr]*my_data_size)
    inr_array = np.array([inr]*my_data_size)
    cons_array = np.array([[cons[i]]*my_data.shape[0] for i in range(0,block)]).reshape(my_data_size,)
    block_array = np.array([([i+1]*my_data.shape[0])for i in range(0, block)]).reshape(my_data_size,)
    label_array = label.T.reshape(my_data_size,)
    test_pd = pd.DataFrame({'real':my_data_div.real,'imag':my_data_div.imag, 'snr':snr_array, 'inr':inr_array,
            'cons':cons_array, 'block':block_array,
            'lreal':label_array.real, 'limag':label_array.imag})
    return test_pd

myTable = table_data(my_data, 0.6, 0.05, my_labels['L_Constellations'][0], X)

'defind training batch'
from numpy.random import default_rng
def training_set(myTable):
    block = myTable.shape[1]
    rng = default_rng()
    sample_size = int(0.8 * block)
    numbers = rng.choice(range(1, block + 1), size=sample_size, replace=False)
    training_dataset = myTable[myTable.block.isin(numbers)]
    return training_dataset


def test_set(myTable, training_dataset):
    remaining = myTable.drop(training_dataset.index)
    return remaining

'set training sample'
train_dataset = training_set(myTable)
test_dataset = test_set(myTable, train_dataset)

'Divided data set'
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = pd.DataFrame([train_features.pop('lreal'),train_features.pop('limag')]).T
test_labels = pd.DataFrame([test_features.pop('lreal'), test_features.pop('limag')]).T

'setup training'
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

def build_and_compile_model(norm):
    model = keras.Sequential([
    norm,
    layers.Dense(50, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(2)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.01))
    return model

dnn_signal_model = build_and_compile_model(normalizer)

history = dnn_signal_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=500)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Ireal]')
    plt.legend()
    plt.grid(True)

test_results = {}
test_results['signal'] = dnn_signal_model.evaluate(
    test_features,
    test_labels, verbose=0)

y = dnn_signal_model.predict(test_features)

def plot_compare(y):
    labels = test_labels.to_numpy();
    plt.scatter(labels[:50,0], labels[:50,1], color='black',label='Data')
    plt.plot(y[:50,0], y[:50,1],  'o', color='red', label='Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

'save report'
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
hist.to_csv('./out.csv', index=False)

'save plot'
plot_compare(y)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

'save model'
dnn_signal_model.save('./my_model')