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
X = my_labels['L_S_x']

def table_data(my_data, cons, label):
    block = my_data.shape[1]
    my_data_size = my_data.shape[0] * block
    my_data_div = my_data.T.reshape(my_data_size,)
    cons_array = np.array([[cons[i]]*my_data.shape[0] for i in range(0,block)]).reshape(my_data_size,)
    block_array = np.array([([i+1]*my_data.shape[0])for i in range(0, block)]).reshape(my_data_size,)
    label_array = label.T.reshape(my_data_size,)
    test_pd = pd.DataFrame({'real':my_data_div.real,'imag':my_data_div.imag,
            'cons':cons_array, 'block':block_array,
            'label':label_array})
    return test_pd
myOrig = table_data(my_data, my_labels['L_Constellations'][0], X)

def assign_labels(myTable):
    myTest = myTable.copy()
    myTest.loc[myTest.cons==2, 'label'] = myTest.loc[myTest.cons==2, 'label']+4
    myTest.label = myTest.label-1
    return myTest
myTable = assign_labels(myOrig)

from numpy.random import default_rng
def training_set(myTable):
    block = myTable.shape[0]
    rng = default_rng()
    sample_size = int(0.8 * block)
    numbers = rng.choice(range(1, block + 1), size=sample_size, replace=False)
    training_dataset = myTable[myTable.block.isin(numbers)]
    return training_dataset

def test_set(myTable, training_dataset):
    remaining = myTable.drop(training_dataset.index)
    return remaining

def test_set(myTable, training_dataset):
    remaining = myTable.drop(training_dataset.index)
    return remaining

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(50, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(20)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(0.01),
                 metrics=['accuracy'])
    return model

def get_total_loss(predict_label, true_label):
    i = 0
    for j in range(len(predict_label)):
        prediction = np.argmax(predict_label[j])
        if prediction != true_label[j]:
            i = i + 1
    return i

train_dataset = training_set(myTable)
test_dataset = test_set(myTable, train_dataset)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = pd.DataFrame([train_features.pop('label')]).T
test_labels = pd.DataFrame([test_features.pop('label')]).T

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
dnn_real_model = build_and_compile_model(normalizer)


history = dnn_real_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
y1 = dnn_real_model.predict(test_features)


print(hist.tail())
print(get_total_loss(y1, test_labels.to_numpy()))