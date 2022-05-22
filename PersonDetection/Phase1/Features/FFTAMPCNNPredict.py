"""
Created by Bing Liu
Human detection using FFT amplitude CNN model
"""
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Header as hd
import os
from FFTAMPCNNet import CNNNet
import Common as com
import datetime

def Run(Test_File, Model_Folder, Freqs_Num, Samples_Num, Model_Name):
    com.Log_Setup(Model_Folder + 'Log/', "Test")
    com.Log_Info('----------------------------------------------------')
    com.Log_Info(str(datetime.datetime.now()))

    msg = "Loading test data from " + Test_File
    com.Log_Info(msg)
    print(msg)
    f_Test = open(Test_File, "rb")
    test_Lables, test_Raws = pickle.load(f_Test)

    X_test = np.asarray(test_Raws, dtype=hd.Raw_Data_Type_Str)
    y_test = np.asarray(test_Lables, dtype=hd.Raw_Data_Type_Str)
    n_test = X_test.shape[0]

    CNNNet.x = tf.placeholder(hd.Raw_Data_Type, (None, Freqs_Num, Samples_Num, 1))
    CNN_Model = CNNNet(n_out = 2)

    msg = "Number of testing examples: {} ".format(n_test)
    com.Log_Info(msg)
    print(msg)
    with tf.Session() as sess:
        msg = "Loading model from " + Model_Folder + Model_Name
        com.Log_Info(msg)
        print(msg)
        CNN_Model.saver.restore(sess, Model_Folder + Model_Name)
        y_pred = CNN_Model.y_predict(X_test)
        test_accuracy = sum(y_test == y_pred)/len(y_test)
        msg = "Test Accuracy = {:.1f}%".format(test_accuracy*100)
        com.Log_Info(msg)
        print(msg)

        predicted  = np.array(y_pred).astype("int")
        actual = np.array(y_test).astype("int")

        TP = np.count_nonzero(predicted * actual)
        TN = np.count_nonzero((predicted - 1) * (actual - 1))
        FP = np.count_nonzero(predicted * (actual - 1))
        FN = np.count_nonzero((predicted - 1) * actual)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        msg_arr = []
        #msg_arr.append("TP = {:d}".format(TP))
        #msg_arr.append("TN = {:d}".format(TN))
        #msg_arr.append("FP = {:d}".format(FP))
        #msg_arr.append("FN = {:d}".format(FN))
        #msg_arr.append("Precision = {:.1f}%".format(precision*100))
        #msg_arr.append("Recall = {:.1f}%".format(recall*100))
        #msg_arr.append("F1 = {:.1f}%".format(f1*100))

        msg_arr.append("{:d}".format(TP))
        msg_arr.append("{:d}".format(TN))
        msg_arr.append("{:d}".format(FP))
        msg_arr.append("{:d}".format(FN))
        msg_arr.append("{:.1f}%".format(precision*100))
        msg_arr.append("{:.1f}%".format(recall*100))
        msg_arr.append("{:.1f}%".format(f1*100))
        com.Log_Info_Arr(msg_arr)
