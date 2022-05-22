"""
Created by Bing Liu
Create dataset to train histogram CNN model
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

def Get_Raw_Data_File_List(Raw_Data_File_Path, Training_File_Location_Filters):
    files_list = []
    data_files = com.Get_File_name_List(Raw_Data_File_Path)
    for file in data_files:
       location_code = com.Get_Location_Code(file, Raw_Data_File_Path)
       for code in Training_File_Location_Filters:
           if code == location_code: 
                files_list.append(file)
                break
    return shuffle(files_list)

def CreateDataSet_Column(Raw_file_list, Reqs_Num):

    Lables = []
    Hists_All = []
    for raw_file in Raw_file_list:
        print(raw_file + "  " + str(com.Get_File_Create_Time(raw_file.replace(hd.Raw_Data_File_Path_Active_Band, hd.Raw_Data_Full_Band))))

        f = open(raw_file, "rb")
        freq_arr = pickle.load(f)
        lable_arr = pickle.load(f)
        Raw_arr = pickle.load(f)
        hist_bins = hd.Hist_Bins
        bins_len =  len(hist_bins)
        
        hist_pack = np.zeros((len(Raw_arr), bins_len-1)).astype(int)

        i = 0
        for raw in Raw_arr:
            hist_pack[i], bin_edges  = np.histogram(raw, bins=hist_bins,  range =(0,255) )
            i = i + 1

        data_arr_temp = np.asarray(hist_pack, dtype=hd.Raw_Data_Type_Str)
        Hist_data_arr_temp = data_arr_temp.reshape(Reqs_Num, bins_len-1, 1)

        Lables.append(lable_arr[0])
        Hists_All.append(Hist_data_arr_temp)

    return Lables, Hists_All

def DumpToFile(Raw_file_list, RawDataSet, Reqs_Num ):
    print("Dump " + str(len(Raw_file_list)) + " files to : " + RawDataSet)
    Lables, Hists = CreateDataSet_Column(shuffle(Raw_file_list), Reqs_Num)
    with open(RawDataSet, 'wb') as f:
        pickle.dump((Lables, Hists), f, pickle.HIGHEST_PROTOCOL)


def Create(Raw_Data_Path, 
           Data_Set_Path, 
           Train, 
           Valid, 
           Test, 
           File_Num_Per_DataSet, 
           Location_Filters,
           Reqs_Num):

    Raw_file_list_all = []
    Raw_file_list_all = Get_Raw_Data_File_List(Raw_Data_Path, Location_Filters)

    Person_Present_Num_No = 0
    Person_Present_Num_Yes = 0
    Raw_File_Num_Pack = 0

    for file in Raw_file_list_all:
        if com.Get_Person_Present_Status(file) == hd.Person_Present_No:
            Person_Present_Num_No = Person_Present_Num_No + 1
        elif com.Get_Person_Present_Status(file) == hd.Person_Present_Yes:
            Person_Present_Num_Yes = Person_Present_Num_Yes + 1

    Raw_File_Num_Pack = Person_Present_Num_No
    if Person_Present_Num_No > Person_Present_Num_Yes:
        Raw_File_Num_Pack = Person_Present_Num_Yes
    
    Raw_file_list_No = []
    Raw_file_list_Yes = []

    for file in Raw_file_list_all:
        if len(Raw_file_list_No) == Raw_File_Num_Pack and len(Raw_file_list_Yes) == Raw_File_Num_Pack:
            break
        if com.Get_Person_Present_Status(file) == hd.Person_Present_No and len(Raw_file_list_No) < Raw_File_Num_Pack:
            Raw_file_list_No.append(file)
        elif com.Get_Person_Present_Status(file) == hd.Person_Present_Yes and len(Raw_file_list_Yes) < Raw_File_Num_Pack:
            Raw_file_list_Yes.append(file)

    test_file_num = int(Raw_File_Num_Pack * 0.05)
    DataSet_Test = Data_Set_Path + Test + hd.Raw_Data_File_Name_Extension
    DumpToFile(Raw_file_list_Yes[0:test_file_num] + Raw_file_list_No[0:test_file_num],
                DataSet_Test,
                Reqs_Num)

    test_file_num_rest = Raw_File_Num_Pack - test_file_num
    File_Num_Per_Set = int(File_Num_Per_DataSet/2)
    DataSet_File_Num = int((test_file_num_rest) / File_Num_Per_Set)

    Raw_file_list_Yes_Rest = []
    Raw_file_list_No_Rest = []

    Raw_file_list_Yes_Rest = Raw_file_list_Yes[test_file_num:Raw_File_Num_Pack]
    Raw_file_list_No_Rest = Raw_file_list_No[test_file_num:Raw_File_Num_Pack]

    for i in range(DataSet_File_Num + 1):
        DataSet_Train = Data_Set_Path + Train + "_" + str(i) + hd.Raw_Data_File_Name_Extension
        DataSet_Valid = Data_Set_Path + Valid + "_" + str(i) + hd.Raw_Data_File_Name_Extension

        start = i * File_Num_Per_Set
        end = (i+1) * File_Num_Per_Set
        if i == DataSet_File_Num:
            end = len(Raw_file_list_Yes_Rest)

        if end - start < File_Num_Per_Set and i >0:
            break

        train_file_num = int((end - start) * 0.8)
        valid_file_num = (end - start) - train_file_num
        DumpToFile(Raw_file_list_Yes_Rest[start: start + train_file_num] + Raw_file_list_No_Rest[start:start + train_file_num],
                   DataSet_Train,
                   Reqs_Num)

        DumpToFile(Raw_file_list_Yes_Rest[start + train_file_num:end] + Raw_file_list_No_Rest[start + train_file_num:end],
                  DataSet_Valid,
                   Reqs_Num)


