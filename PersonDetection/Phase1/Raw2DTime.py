"""
Created by Bing Liu
Human detection in different time frame
"""
import Common as com
import tensorflow as tf
import pickle
import numpy as np
import Header as hd
import Raw2DCreateDataSet
import Raw2DCNNTrain
import Raw2DCNNPredict
from Raw2DCNNNet import CNNNet
import datetime

Model_Name = "CNN"
Create_Dataset = False
Training = False
Predict = True

Raw_Data_File_Path_Full_Band = hd.Raw_Data_Full_Band
Raw_Data_File_Path = hd.Raw_Data_File_Path_Active_Band
Data_Set_Path = hd.Raw2D_Data_Set_Path_Time
Model_Path = hd.Raw2D_Model_Path_Time

Freqs_Num = len(com.Load_Feq_List(hd.Root + hd.Freq_List_File_Active_Band ))

if Create_Dataset == True:
    com.Remove_Data_File(Data_Set_Path, hd.Raw_Data_File_Name_Extension)
    Raw2DCreateDataSet.CreateByTime(Raw_Data_File_Path,
                              Raw_Data_File_Path_Full_Band,
                              Data_Set_Path,
                              hd.Raw2D_Data_Set_Train,
                              hd.Raw2D_Data_Set_Valid,
                              hd.Raw2D_Data_Set_Test,
                              300,
                              ["01"],
                              Freqs_Num,
                              hd.Samples_Num)
if Training == True:

    EPOCHS = 10
    BATCH_SIZE = 16
    Raw2DCNNTrain.Run(EPOCHS,
                      BATCH_SIZE,
                      Model_Path,
                      Data_Set_Path,
                      hd.Raw2D_Data_Set_Train,
                      hd.Raw2D_Data_Set_Valid,
                      hd.Raw2D_Data_Set_Test,
                      Freqs_Num,
                      hd.Samples_Num,
                      Model_Name,
                      Run_Test = False)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if Predict == True:

    com.Log_Setup(Model_Path + "Log/", "Test")
    msg_arr = []
    msg_arr.append('----------------------------------------------------')
    msg_arr.append(str(datetime.datetime.now()))

    Hours_Per_Period = int(24/hd.Time_Period_Num)
    CNNNet.x = tf.placeholder(hd.Raw_Data_Type, (None, Freqs_Num, hd.Samples_Num, 1))
    CNN_Model = CNNNet(n_out = 2)

    with tf.Session() as sess:
        msg = "Loading model from " + Model_Path + Model_Name
        print(msg)
        msg_arr.append(msg)

        CNN_Model.saver.restore(sess, Model_Path + Model_Name)

        for i in range(hd.Time_Period_Num):
            Test_File = Data_Set_Path + hd.Raw2D_Data_Set_Test + "_" + str(i) + hd.Raw_Data_File_Name_Extension
            Hour_Start = i*Hours_Per_Period
            Hour_End = (i + 1) * Hours_Per_Period - 1
            msg = "Loading model from " + Model_Path + Model_Name
            print(msg)
            msg_arr.append(msg)

            #print("Hour from: " + str(Hour_Start) + " to : " + str(Hour_End))
            msg = "Hour from:  {0:d}  to : {1:d}".format(Hour_Start, Hour_End)
            print(msg)
            msg_arr.append(msg)

            msg = "Loading test data from " + Test_File
            print(msg)
            msg_arr.append(msg)

            f_Test = open(Test_File, "rb")
            test_Lables, test_Raws = pickle.load(f_Test)

            X_test = np.asarray(test_Raws, dtype=hd.Raw_Data_Type_Str)
            y_test = np.asarray(test_Lables, dtype=hd.Raw_Data_Type_Str)
            n_test = X_test.shape[0]
            msg = "Number of testing examples: {0:d}".format(n_test)
            print(msg)
            msg_arr.append(msg)

            y_pred = CNN_Model.y_predict(X_test)
            test_accuracy = sum(y_test == y_pred)/len(y_test)
            msg = "Test Accuracy = {:.1f}%".format(test_accuracy*100)
            print(msg)
            msg_arr.append(msg)

            predicted  = np.array(y_pred).astype("int")
            actual = np.array(y_test).astype("int")

            TP = np.count_nonzero(predicted * actual)
            TN = np.count_nonzero((predicted - 1) * (actual - 1))
            FP = np.count_nonzero(predicted * (actual - 1))
            FN = np.count_nonzero((predicted - 1) * actual)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)

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
    msg_arr.clear()
