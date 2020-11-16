import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc
import pandas as pd
import mlcTest as mt

dataFile = "./16QAM/my_data.mat"
labelFile = './16QAM/my_label.mat'
my_data = sc.loadmat(dataFile)
my_labels = sc.loadmat(labelFile)

def dataClean(my_data, my_labels):
    my_sample = my_data['y'].T
    my_label = my_labels['ip'].T
    array = [-3,-1,1,3]
    symbols = [complex(i,j)for i in array for j in array]
    a = []
    for i in range((my_label.shape[0])):
        for j in range(len(symbols)):
            if my_label[i] == symbols[j]:
                a.append(j)
    size = my_sample.shape[0]
    my_Table = pd.DataFrame({'real': my_sample.real.reshape(size), 'imag': my_sample.imag.reshape(size),
                  'label': a})
    return my_Table

if __name__ == "__main__":

    dataFile = "./16QAM/my_data.mat"
    labelFile = './16QAM/my_label.mat'
    my_data = sc.loadmat(dataFile)
    my_labels = sc.loadmat(labelFile)
    table = dataClean(my_data, my_labels)
    cf_ori, result = mt.get_train(table, 1)
    print(cf_ori, "\n"
        "result is", result)