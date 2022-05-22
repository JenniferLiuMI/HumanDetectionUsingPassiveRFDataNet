"""
Created by Bing Liu
Create dataset used to train FFT amplitude CNN model
"""
import Common as com
import Header as hd
import pickle
import threading
import numpy as np
from time import sleep
from sklearn.utils import shuffle
import sys
np.set_printoptions(threshold=sys.maxsize)
from sys import maxsize
from numpy import set_printoptions

set_printoptions(threshold=maxsize)

def Get_Data_File_List(FFT_Data_File_Path, Training_File_Location_Filters):
    files_list = []
    data_files = com.Get_File_name_List(FFT_Data_File_Path)
    for file in data_files:
       location_code = Get_Location_Code(file, FFT_Data_File_Path)
       for code in Training_File_Location_Filters:
           if code == location_code: 
                files_list.append(file)
                break
    return shuffle(files_list)

def Get_Location_Code(file_name, FFT_Data_File_Path):
    index_S = len(FFT_Data_File_Path) + len(hd.FFT_Data_File_Name_Prefix) + len(hd.Raw_Data_File_Name_Start_Index) + 1
    len_location_code = len(hd.Colect_Location_Code_StartIndex)
    location_code = file_name[index_S: index_S + len_location_code] 
    return location_code

def CreateDataSet_Column(FFT_file_list, Reqs_Num):

    Lables = []
    FFT_AMP_All = []
    IQ_Num = int(hd.Samples_Num/2)
    for raw_file in FFT_file_list:
        print(raw_file + "  " + str(com.Get_File_Create_Time(raw_file.replace(hd.Raw_Data_File_Path_Active_Band, hd.Raw_Data_Full_Band))))

        f = open(raw_file, "rb")
        freq_arr = pickle.load(f)
        lable_arr = pickle.load(f)
        FFT_arr = pickle.load(f)
        fft_amp_pack =  np.absolute(FFT_arr)
        fft_amp_pack = fft_amp_pack.reshape(len(fft_amp_pack), IQ_Num, 1)
        Lables.append(lable_arr[0])
        FFT_AMP_All.append(fft_amp_pack)

    return Lables, FFT_AMP_All


def DumpToFile(fff_file_list, FFTAMPDataSet, Reqs_Num):
    print("Dump " + str(len(fff_file_list)) + " files to : " + FFTAMPDataSet)
    Lables, Values = CreateDataSet_Column(shuffle(fff_file_list), Reqs_Num)
    with open(FFTAMPDataSet, 'wb') as f:
        pickle.dump((Lables, Values), f, pickle.HIGHEST_PROTOCOL)


def Create(FFT_Data_Path, 
           Data_Set_Path, 
           Train, 
           Valid, 
           Test, 
           File_Num_Per_DataSet, 
           Location_Filters,
           Reqs_Num,):

    file_list_all = []
    file_list_all = Get_Data_File_List(FFT_Data_Path, Location_Filters)

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
    
    File_list_No = []
    File_list_Yes = []

    for file in file_list_all:
        if len(File_list_No) == File_Num_Pack and len(File_list_Yes) == File_Num_Pack:
            break
        if com.Get_Person_Present_Status(file) == hd.Person_Present_No and len(File_list_No) < File_Num_Pack:
            File_list_No.append(file)
        elif com.Get_Person_Present_Status(file) == hd.Person_Present_Yes and len(File_list_Yes) < File_Num_Pack:
            File_list_Yes.append(file)

    test_file_num = int(File_Num_Pack * 0.05)
    DataSet_Test = Data_Set_Path + Test + hd.Raw_Data_File_Name_Extension
    DumpToFile(File_list_Yes[0:test_file_num] + File_list_No[0:test_file_num],
                DataSet_Test,
                Reqs_Num)

    test_file_num_rest = File_Num_Pack - test_file_num
    File_Num_Per_Set = int(File_Num_Per_DataSet/2)
    DataSet_File_Num = int((test_file_num_rest) / File_Num_Per_Set)

    File_list_Yes_Rest = []
    File_list_No_Rest = []

    File_list_Yes_Rest = File_list_Yes[test_file_num:File_Num_Pack]
    File_list_No_Rest = File_list_No[test_file_num:File_Num_Pack]

    for i in range(DataSet_File_Num + 1):
        DataSet_Train = Data_Set_Path + Train + "_" + str(i) + hd.Raw_Data_File_Name_Extension
        DataSet_Valid = Data_Set_Path + Valid + "_" + str(i) + hd.Raw_Data_File_Name_Extension

        start = i * File_Num_Per_Set
        end = (i+1) * File_Num_Per_Set
        if i == DataSet_File_Num:
            end = len(File_list_Yes_Rest)

        if end - start < File_Num_Per_Set and i >0:
            break

        train_file_num = int((end - start) * 0.8)
        valid_file_num = (end - start) - train_file_num
        DumpToFile(File_list_Yes_Rest[start: start + train_file_num] + File_list_No_Rest[start:start + train_file_num],
                   DataSet_Train,
                   Reqs_Num)

        DumpToFile(File_list_Yes_Rest[start + train_file_num:end] + File_list_No_Rest[start + train_file_num:end],
                  DataSet_Valid,
                   Reqs_Num)


