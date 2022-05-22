"""
Created by Bing Liu
Traing IQ CNN model
"""
import pickle
import numpy as np
from random import shuffle
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime
import Header as hd
from IQCNNet import CNNNet
import Common as com
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def Run(EPOCHS,
        BATCH_SIZE,
        Model_Folder,
        Data_Set_Path,
        Train_File_Name,
        Valid_File_Name,
        Test_File_Name,
        Freqs_Num,
        Samples_Num,
        Model_Name,
        Run_Test = False):
    msg_arr = []
    logdir = Model_Folder + 'Log'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)
    com.Log_Setup_Train(Model_Folder)
    msg_arr.append('----------------------------------------------------')
    msg_arr.append(str(datetime.datetime.now()))

    #Timer start
    train_start = datetime.datetime.now()
 
    train_file_list = []
    train_file_list = com.Get_File_List(Data_Set_Path, Train_File_Name);

    valid_file_list = []
    valid_file_list = com.Get_File_List(Data_Set_Path, Valid_File_Name);

    n_train_file = len(train_file_list)

    print("Number of classes =", CNNNet.n_classes)
    
    CNNNet.x = tf.placeholder(hd.Raw_Data_Type, (None, Freqs_Num, Samples_Num, 2))
    CNN_Model = CNNNet(n_out = CNNNet.n_classes)

    Train_Writer_Index = 0
    Valid_Writer_Index = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training...")

        Train_Writer = tf.summary.FileWriter(logdir + '/Training', sess.graph)
        Valid_Writer = tf.summary.FileWriter(logdir + '/Validation', sess.graph)

        for i in range(EPOCHS):
            msg = "EPOCH {}".format(i+1)
            print(msg)
            msg_arr.append(msg)

            for j in range(n_train_file):

                Train = Data_Set_Path + Train_File_Name + "_" + str(j) + hd.Raw_Data_File_Name_Extension
                f_Train = open(Train, "rb")
                train_Lables, train_Raws = pickle.load(f_Train)

                Valid = Data_Set_Path + Valid_File_Name + "_" + str(j) + hd.Raw_Data_File_Name_Extension
                f_Valid = open(Valid, "rb")
                valid_Lables, valid_Raws = pickle.load(f_Valid)

                x_train = np.asarray(train_Raws, dtype=hd.Raw_Data_Type_Str)
                y_train = np.asarray(train_Lables, dtype=hd.Raw_Data_Type_Str)

                x_valid = np.asarray(valid_Raws, dtype=hd.Raw_Data_Type_Str)
                y_valid = np.asarray(valid_Lables, dtype=hd.Raw_Data_Type_Str)

                n_train = x_train.shape[0]
                n_validation = x_valid.shape[0]

                msg = "Training dataset {}".format(j)
                print(msg)
                msg_arr.append(msg)

                msg = "Number of training examples: {}".format(n_train)
                print(msg)
                msg_arr.append(msg)

                msg = "Number of validation examples: {}".format(n_validation)
                print(msg)
                msg_arr.append(msg)

                x_train, y_train = shuffle(x_train, y_train)
                num_examples = len(y_train)

                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                    summary, _ = sess.run([CNN_Model.merged, CNN_Model.training_operation],
                                         feed_dict={CNNNet.x: batch_x, CNNNet.y: batch_y, CNNNet.keep_prob : 0.5, CNNNet.keep_prob_conv: 0.7})
                    Train_Writer.add_summary(summary, Train_Writer_Index)
                    Train_Writer_Index = Train_Writer_Index + 1

                Valid_Writer_Index, validation_accuracy = CNN_Model.evaluate(x_valid, y_valid, Valid_Writer, Valid_Writer_Index)
                msg = "Validation Accuracy = {:.3f}%".format(validation_accuracy*100)
                msg_arr.append(msg)
                print(msg)
        CNN_Model.saver.save(sess, os.path.join(Model_Folder, Model_Name))
        msg = "Model saved: " + os.path.join(Model_Folder, Model_Name)
        msg_arr.append(msg)
        print(msg)
        #Timer end
        train_end = datetime.datetime.now()
        Time_Diff = train_end - train_start
        msg = "Training duration:" + str(Time_Diff)
        msg_arr.append(msg)
        print(msg)

    if Run_Test == True:
        Test = Data_Set_Path + Test_File_Name + hd.Raw_Data_File_Name_Extension
        f_Test = open(Test, "rb")
        test_Lables, test_Raws = pickle.load(f_Test)

        X_test = np.asarray(test_Raws, dtype=hd.Raw_Data_Type_Str)
        y_test = np.asarray(test_Lables, dtype=hd.Raw_Data_Type_Str)
        n_test = X_test.shape[0]

        msg = "Number of testing examples: {} ".format(n_test)
        msg_arr.append(msg)
        print(msg)

        with tf.Session() as sess:
            msg = "Loading model from " + Model_Folder + Model_Name
            msg_arr.append(msg)
            print(msg)
            CNN_Model.saver.restore(sess, Model_Folder + Model_Name)
            y_pred = CNN_Model.y_predict(X_test)
            test_accuracy = sum(y_test == y_pred)/len(y_test)

            msg = "Test Accuracy = {:.1f}%".format(test_accuracy*100)
            msg_arr.append(msg)
            print(msg)
    com.Log_Info_Arr(msg_arr)
    msg_arr.clear()
