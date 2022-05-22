"""
Created by Bing Liu
Functions to process power data
"""
import pickle
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
sys.path.insert(1, '/media/win/Code/RFResearch/PersonDetection')
#sys.path.insert(1, 'E:\Code\RFResearch\PersonDetection')
import Header as hd
import time
import Common as com
sys.path.insert(1, '/media/win/Code/RFResearch/PersonDetection/Phase1')
#sys.path.insert(1, 'E:\Code\RFResearch\PersonDetection\Phase1')
import Raw2DCreateDataSet
import Com_Power
from sklearn.utils import shuffle

GAN_Power_Avg_Data_Set_Path_01 = hd.GAN_Power_Avg_Data_Set_Path_01
file_list_all_01 = Com_Power.Get_Power_Avg_File_List_Location_01_Hour_From_6_To_22()

GAN_Power_Avg_Data_Set_Path_05 = hd.GAN_Power_Avg_Data_Set_Path_05
file_list_all_05 = Com_Power.Get_Power_Avg_File_List_Location_05_Hour_From_6_To_22()

Phase2_root = '/media/win/RFRoot/Phase2/'
GAN_Power_Avg_Data_Set_Path_Phase2 = Phase2_root + 'Power/'

#GAN_Power_Avg_Data_Set_Path = hd.GAN_Power_Avg_Data_Set_Path_01_05
#file_list_all = Com_Power.Get_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22()
#random.shuffle(file_list_all)

def DumpToFile(Power_Avg_File_List, DataSet):
    print("Dump " + str(len(Power_Avg_File_List)) + " files to : " + DataSet)
    file_num = len(Power_Avg_File_List)

    freq_arr, lables_per_file, freq_powers_per_file = pickle.load(open(Power_Avg_File_List[0], "rb"))
    label_arr = np.zeros(file_num, dtype=np.int)
    power_avg_arr = np.zeros((file_num,len(freq_powers_per_file)), dtype=np.float)

    for i in range(file_num):
        freq_arr, lables_per_file, freq_powers_per_file = pickle.load(open(Power_Avg_File_List[i], "rb"))
        label_arr[i] = lables_per_file[0]
        power_avg_arr[i] = freq_powers_per_file

    with open(DataSet, 'wb') as f:
        pickle.dump((freq_arr,label_arr,power_avg_arr), f, pickle.HIGHEST_PROTOCOL)

def create():

    Person_Present_Num_No = 0
    Person_Present_Num_Yes = 0
    File_Num_Pack = 0

    for file in file_list_all:
        if com.Get_Person_Present_Status(file) == hd.Person_Present_No:
            Person_Present_Num_No = Person_Present_Num_No + 1
        elif com.Get_Person_Present_Status(file) == hd.Person_Present_Yes:
            Person_Present_Num_Yes = Person_Present_Num_Yes + 1

    File_Num_Pack = Person_Present_Num_No
    if Person_Present_Num_No > Person_Present_Num_Yes:
        File_Num_Pack = Person_Present_Num_Yes
    
    file_list_No = []
    file_list_Yes = []

    for file in file_list_all:
        if len(file_list_No) == File_Num_Pack and len(file_list_Yes) == File_Num_Pack:
            break
        if com.Get_Person_Present_Status(file) == hd.Person_Present_No and len(file_list_No) < File_Num_Pack:
            file_list_No.append(file)
        elif com.Get_Person_Present_Status(file) == hd.Person_Present_Yes and len(file_list_Yes) < File_Num_Pack:
            file_list_Yes.append(file)

    train_file_num = int(File_Num_Pack * 0.7)
    DataSet = GAN_Power_Avg_Data_Set_Path + hd.Data_Set_Train + hd.Raw_Data_File_Name_Extension
    DumpToFile(file_list_Yes[0:train_file_num] + file_list_No[0:train_file_num],
                DataSet)

    valid_file_num = int(File_Num_Pack * 0.25)
    DataSet = GAN_Power_Avg_Data_Set_Path + hd.Data_Set_Valid + hd.Raw_Data_File_Name_Extension
    DumpToFile(file_list_Yes[train_file_num:train_file_num+valid_file_num] + file_list_No[train_file_num:train_file_num+valid_file_num],
                DataSet)

    test_file_num = File_Num_Pack - train_file_num - valid_file_num
    DataSet = GAN_Power_Avg_Data_Set_Path + hd.Data_Set_Test + hd.Raw_Data_File_Name_Extension
    DumpToFile(file_list_Yes[-test_file_num:-1] + file_list_No[-test_file_num:-1],
                DataSet)

def load(Select_Band_Num, figure_dir, location):

    freqs_selected = np.array(com.Import_From_Text_To_Int(hd.PCA_Selected_Band_List_File.format(Select_Band_Num)))
    freqs_selected = np.sort(freqs_selected)

    if location == '01':
        DataSet_name_train = GAN_Power_Avg_Data_Set_Path_01 + hd.Data_Set_Train + hd.Raw_Data_File_Name_Extension
        DataSet_name_valid = GAN_Power_Avg_Data_Set_Path_01 + hd.Data_Set_Valid + hd.Raw_Data_File_Name_Extension
        freq_arr_all,y_train,x_train = pickle.load(open(DataSet_name_train, "rb"))
        freq_arr_all,y_valid,x_valid = pickle.load(open(DataSet_name_valid, "rb"))
    elif location == '05':
        DataSet_name_train = GAN_Power_Avg_Data_Set_Path_05 + hd.Data_Set_Train + hd.Raw_Data_File_Name_Extension
        DataSet_name_valid = GAN_Power_Avg_Data_Set_Path_05 + hd.Data_Set_Valid + hd.Raw_Data_File_Name_Extension
        freq_arr_all,y_train,x_train = pickle.load(open(DataSet_name_train, "rb"))
        freq_arr_all,y_valid,x_valid = pickle.load(open(DataSet_name_valid, "rb"))
    else :
        DataSet_name_Yes = os.path.join( GAN_Power_Avg_Data_Set_Path_Phase2,  'Power_All_Yes_{}.pkl'.format(location))
        DataSet_name_No = os.path.join( GAN_Power_Avg_Data_Set_Path_Phase2,  'Power_All_No_{}.pkl'.format(location))
        x_all_Yes = pickle.load(open(DataSet_name_Yes, "rb"))
        x_all_No = pickle.load(open(DataSet_name_No, "rb"))
        y_all_Yes = np.ones(len(x_all_Yes), dtype=np.int16)
        y_all_No = np.zeros(len(x_all_No), dtype=np.int16)
        x_all = np.concatenate((x_all_Yes, x_all_No), axis=0)
        y_all = np.concatenate((y_all_Yes, y_all_No), axis=0)
        
        seed = int((time.time()- int(time.time()))*10000)
        np.random.seed(seed)
        np.random.shuffle(x_all)
        np.random.seed(seed)
        np.random.shuffle(y_all)
                
        x_train = x_all[:int(len(x_all)*0.8)]
        x_valid = x_all[int(len(x_all)*0.8):]
        y_train = y_all[:int(len(x_all)*0.8)]
        y_valid = y_all[int(len(x_all)*0.8):]
        freq_arr_all_file = os.path.join( Phase2_root,  'FreqFullBand.txt')
        freq_arr_all = com.Import_From_Text_To_Int(freq_arr_all_file)


    indices = np.in1d(freq_arr_all, freqs_selected).nonzero()[0]  
    x_train_select_band = np.take(x_train, indices, axis=1)
    x_valid_select_band = np.take(x_valid, indices, axis=1)

    figure_name = os.path.join(figure_dir, "Train_Power_{}".format(location) + hd.JPG_File_Extension)
    Com_Power.Draw_Freqs_Powers_Avg_Train_By_Power(figure_name, freqs_selected, x_train_select_band, y_train, x_valid_select_band, y_valid)
    
    figure_name = os.path.join(figure_dir, "Train_Power_Sample_{}".format(location) + hd.JPG_File_Extension)
    Com_Power.Draw_Freqs_Figure_Single_Sub_Plot(figure_name, freqs_selected, x_train_select_band[:16], 4, 4)

    return freqs_selected, x_train_select_band/(-40.0), y_train, x_valid_select_band/(-40.0), y_valid
#load(40, '/media/win/Code/RFResearch/GANout/Power_Avg/')
#create()
