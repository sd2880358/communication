import tensorflow as tf
import numpy as np
'import matplotlib.pyplot as plt'
import scipy.io as sc
import pandas as pd
'import seaborn as sns'
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


def get_training(myTable, epochs, cross_data):
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

    cf_ori = confusion_matrix(probability_model, test_features, test_labels)
    cross_features = cross_data.copy()
    cross_label  = pd.DataFrame([cross_features.pop('label')]).T
    cf_cross = confusion_matrix(probability_model, cross_features, cross_label)
    test_results['cross'] = dnn_real_model.evaluate(
        cross_features,
        cross_label, verbose=0
    )
    '''
    hist['epoch'] = history.epoch
    result1 = hist.tail()
    result1 = result1.append(test_results, ignore_index=True)
    hist.to_csv('./result/'+ test_time + '/' + files + '.csv', index=False)
    '''
    return [cf_ori, cf_cross, test_results]

def confusion_matrix(model, feature, label):
    expected = model.predict(feature)
    expected = get_results(expected)
    cf = tf.math.confusion_matrix(expected, label.to_numpy()).numpy()
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
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_heatmap(ax1, symbol_error_results_4, cons_4_error)
    plot_heatmap(ax2, symbol_error_results_16, cons_16_error)
    plt.savefig("./result/"+ test_time+'/'+file_name, dpi=500)
    '''
def batch_result(cf):
    array = []
    cons_4 = cf[:4]
    cons_16 = cf[4:]
    for j in range(len(cf)):
        result = cf.iloc[j, j] / cf.iloc[j, :].sum()
        array.append(result)
    cons_4_error = (cons_4.iloc[:, 4:].sum().sum()) / cons_4.sum().sum()
    cons_16_error = (cons_16.iloc[:, :4].sum().sum()) / cons_4.sum().sum()
    array.append(cons_4_error)
    array.append(cons_16_error)
    array = pd.DataFrame(array).T
    return array

def cross_test(train_set, test_set, ori, cross, test_result):
    test_dataset = test_set.sample(frac=0.2, random_state=0)
    cf_ori, cf_cross, result = get_training(train_set, 20, test_dataset)
    cf_ori_array = batch_result(cf_ori)
    cf_cross_array = batch_result(cf_cross)
    result = pd.DataFrame(result).T
    ori = ori.append(cf_ori_array)
    cross = cross.append(cf_cross_array)
    test_result = test_result.append(result)
    return [ori, cross, test_result]



data1 = "intermediate"
data1_label = "intermediate_label"
data2 = "hard"
data2_label = "hard_label"
data3 = "data"
data3_label = "data_label"

table1 = dataset(data1, data1_label)
table2 = dataset(data2, data2_label)
test1_ori = pd.DataFrame()
test2_ori = pd.DataFrame()
test1_cross = pd.DataFrame()
test2_cross = pd.DataFrame()
test1_accuracy = pd.DataFrame()
test2_accuracy =  pd.DataFrame()
file_directory = './result/cross_testing'


for i in range(0,10):
    i = str(i+1)
    test1_ori, test1_cross, test1_accuracy = cross_test(table1, table2, test1_accuracy)
    test2_ori, test2_cross, test2_accuracy = cross_test(table2, table1, test2_accuracy)
    print("This is the {} time".format(i))



test1_ori.to_csv(file_directory +'/test1_ori_error_rate.csv', index=False)
test1_cross.to_csv(file_directory +'/test1_cross_error_rate.csv', index=False)
test2_ori.to_csv(file_directory + '/test2_ori_error_rate.csv', index=False)
test2_cross.to_csv(file_directory + '/test2_cross_error_rate.csv', index=False)
test1_accuracy.to_csv(file_directory + '/test1_accuracy.csv', index=False)
test2_accuracy.to_csv(file_directory + '/test1_accuracy.csv', index=False)

