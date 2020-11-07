import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing




def dataset(dataFile, labelFile):
    dataFile = "./communication/" + dataFile
    labelFile = "./communication/" + labelFile
    my_data = sc.loadmat(dataFile)
    my_labels = sc.loadmat(labelFile)
    my_data = my_data['Y']
    X = my_labels['L_S_x']
    myOrig = table_data(my_data, my_labels['L_Constellations'][0], X)
    myTable = assign_labels(myOrig)
    return myTable


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


def assign_labels(myTable):
    myTest = myTable.copy()
    myTest.loc[myTest.cons == 2, 'label'] = myTest.loc[myTest.cons == 2, 'label'] + 4
    myTest.label = myTest.label - 1
    return myTest


from numpy.random import default_rng


def training_set(myTable):
    block = myTable.shape[0]
    rng = default_rng()
    sample_size = int(0.8 * block)
    numbers = rng.choice(range(1, block + 1), size=sample_size, replace=False)
    training_dataset = myTable[myTable.block.isin(numbers)]
    return training_dataset

def get_results(results):
    prediction_results = []
    for i in range(len(results)):
        prediction_results.append(np.argmax(results[i]))
    return prediction_results

def test_set(myTable, training_dataset):
    remaining = myTable.drop(training_dataset.index)
    return remaining


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(20)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])
    return model

'''
def get_total_loss(predict_label, true_label):
    i = 0
    for j in range(len(predict_label)):
        prediction = np.argmax(predict_label[j])
        if prediction != true_label[j]:
            i = i + test1
    return i / len(predict_label)
'''


def get_training(myTable, epochs, files, test_time):
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
        epochs=epochs, verbose=0)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    test_results = {}
    test_results['signal'] = dnn_real_model.evaluate(
        test_features,
        test_labels, verbose=0)
    probability_model = tf.keras.Sequential([dnn_real_model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_features)
    predicted_results = get_results(predictions)
    cf = tf.math.confusion_matrix(predicted_results, test_labels.to_numpy()).numpy()
    cf = pd.DataFrame(cf)
    hist['epoch'] = history.epoch
    hist.tail()
    hist = hist.append(test_results, ignore_index=True)
    hist.to_csv('./result/'+ test_time + '/' + files + '.csv', index=False)
    return cf

def plot_heatmap(ax, symbol_error, cons_error):
    df = sns.heatmap(symbol_error, annot=True,cmap=plt.cm.Blues, ax=ax)
    df.xaxis.tick_top()
    df.set_xlabel(['P(cons error)= {:.2f}'.format(cons_error)])

'''
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Ireal]')
    plt.legend()
    plt.grid(True)
'''

def prediction(model, testTable, test_results, fileNanme):
    test_features = testTable.copy()
    test_labels = pd.DataFrame([test_features.pop('label')]).T
    test_results[fileNanme] = model.evaluate(
        test_features,
        test_labels, verbose=0)

def divide_Result(cf, file_name, test_time):
    cons_4 = cf[:4]
    cons_16 = cf[4:]
    cons_4_error = (cons_4.iloc[:, 4:].sum().sum()) / cons_4.sum().sum()
    cons_16_error = (cons_16.iloc[:, :4].sum().sum()) / cons_4.sum().sum()
    symbol_error_results_4 = np.array([1 - cons_4.iloc[i, i] /
                                       cons_4.iloc[i, :].sum() for i in range(4)]).reshape(2, 2)
    symbol_error_results_16 = np.array(
        [1 - cons_16.iloc[i - 4, i] /
         cons_16.iloc[i - 4, :].sum() for i in range(4, 20)]).reshape(4, 4)
    symbol_error_results_4 = pd.DataFrame(symbol_error_results_4)
    symbol_error_results_16 = pd.DataFrame(symbol_error_results_16)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_heatmap(ax1, symbol_error_results_4, cons_4_error)
    plot_heatmap(ax2, symbol_error_results_16, cons_16_error)
    plt.savefig("./result/"+ test_time+'/'+file_name, dpi=500)

data1 = "data1"
data1_label = "data1_label"
data2 = "data2"
data2_label = "data2_label"
data3 = "data3"
data3_label = "data3_label"

table1 = dataset(data1, data1_label)
table2 = dataset(data2, data2_label)
table3 = dataset(data3, data3_label)

test = [table1, table2, table3]
name = [data1, data2, data3]
time = "test_2"
for i in range(0,10):
    i = str(i)
    for j in range(len(test)):
        test_results = get_training(test[j], 10, name[j]+i, time)
        divide_Result(test_results, name[j]+i, time)
    print("this is the {} time;".format(i))
