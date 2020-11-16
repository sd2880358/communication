import mlcTest as mt
import pandas as pd
import os
data1 = "intermediate"
data1_label = "intermediate_label"
data2 = "hard"
data2_label = "hard_label"
data3 = "data"
data3_label = "data_label"

table1 = mt.dataset(data1, data1_label)
table2 = mt.dataset(data2, data2_label)
test1_ori = pd.DataFrame()
test2_ori = pd.DataFrame()
test1_cross = pd.DataFrame()
test2_cross = pd.DataFrame()
test1_accuracy = pd.DataFrame()
test2_accuracy =  pd.DataFrame()
file_directory = './result/cross_testing'


for i in range(0,10):
    i = str(i+1)
    test1_ori, test1_cross, test1_accuracy = mt.cross_test(table1, table2,
                                                        test1_ori, test1_cross,
                                                        test1_accuracy)
    test2_ori, test2_cross, test2_accuracy = mt.cross_test(table2, table1,
                                                        test2_ori, test2_cross,
                                                        test2_accuracy)
    print("This is the {} time".format(i))



if not os.path.exists(file_directory):
    os.makedirs(file_directory)
test1_ori.to_csv(file_directory +'/test1_ori_error_rate.csv', index=False)
test1_cross.to_csv(file_directory +'/test1_cross_error_rate.csv', index=False)
test2_ori.to_csv(file_directory + '/test2_ori_error_rate.csv', index=False)
test2_cross.to_csv(file_directory + '/test2_cross_error_rate.csv', index=False)
test1_accuracy.to_csv(file_directory + '/test1_accuracy.csv', index=False)
test2_accuracy.to_csv(file_directory + '/test2_accuracy.csv', index=False)