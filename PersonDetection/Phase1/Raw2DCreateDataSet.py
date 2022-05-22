"""
Created by Bing Liu
Creat dataset
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

def CreateDataSet_Column(Raw_file_list, Reqs_Num, Samples_Num):

    Lables = []
    Raws_All = []
    for raw_file in Raw_file_list:
        print(raw_file + "  " + str(com.Get_File_Create_Time(raw_file.replace(hd.Raw_Data_File_Path_Active_Band, hd.Raw_Data_Full_Band))))

        f = open(raw_file, "rb")
        freq_arr = pickle.load(f)
        lable_arr = pickle.load(f)
        Raw_arr = pickle.load(f)

        data_arr_temp = np.asarray(Raw_arr, dtype=hd.Raw_Data_Type_Str)
        Raw_data_arr_temp = data_arr_temp.reshape(Reqs_Num, Samples_Num, 1)

        Lables.append(lable_arr[0])
        Raws_All.append(Raw_data_arr_temp)

    return Lables, Raws_All

def DumpToFile(Raw_file_list, RawDataSet, Reqs_Num, Samples_Num):
    print("Dump " + str(len(Raw_file_list)) + " files to : " + RawDataSet)
    Lables, Raws = CreateDataSet_Column(shuffle(Raw_file_list), Reqs_Num, Samples_Num)
    with open(RawDataSet, 'wb') as f:
        pickle.dump((Lables, Raws), f, pickle.HIGHEST_PROTOCOL)


def Create(Raw_Data_Path, 
           Raw2D_Data_Set_Path, 
           Raw2D_Train, 
           Raw2D_Valid, 
           Raw2D_Test, 
           Raw2D_File_Num_Per_DataSet, 
           Training_File_Location_Filters,
           Reqs_Num,
           Samples_Num):

    Raw_file_list_all = []
    #Raw_file_list_all = Get_Raw_Data_File_List(Raw_Data_Path, Training_File_Location_Filters)
    file = open(hd.GAN_Power_Avg_File_List_Location_01_Hour_From_6_To_22_Raw, "rb")
    Raw_file_list_all = pickle.load(file)
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
    DataSet_Test = Raw2D_Data_Set_Path + Raw2D_Test + hd.Raw_Data_File_Name_Extension
    DumpToFile(Raw_file_list_Yes[0:test_file_num] + Raw_file_list_No[0:test_file_num],
                DataSet_Test,
                Reqs_Num,
                Samples_Num)

    test_file_num_rest = Raw_File_Num_Pack - test_file_num
    File_Num_Per_Set = int(Raw2D_File_Num_Per_DataSet/2)
    DataSet_File_Num = int((test_file_num_rest) / File_Num_Per_Set)

    Raw_file_list_Yes_Rest = []
    Raw_file_list_No_Rest = []

    Raw_file_list_Yes_Rest = Raw_file_list_Yes[test_file_num:Raw_File_Num_Pack]
    Raw_file_list_No_Rest = Raw_file_list_No[test_file_num:Raw_File_Num_Pack]

    for i in range(DataSet_File_Num + 1):
        DataSet_Train = Raw2D_Data_Set_Path + Raw2D_Train + "_" + str(i) + hd.Raw_Data_File_Name_Extension
        DataSet_Valid = Raw2D_Data_Set_Path + Raw2D_Valid + "_" + str(i) + hd.Raw_Data_File_Name_Extension

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
                   Reqs_Num,
                   Samples_Num)

        DumpToFile(Raw_file_list_Yes_Rest[start + train_file_num:end] + Raw_file_list_No_Rest[start + train_file_num:end],
                  DataSet_Valid,
                   Reqs_Num,
                   Samples_Num)


def CreateByTime(Raw_Data_Path, 
           Path_Full_Band,
           Raw2D_Data_Set_Path, 
           Raw2D_Train, 
           Raw2D_Valid, 
           Raw2D_Test, 
           Raw2D_File_Num_Per_DataSet, 
           Training_File_Location_Filters,
           Reqs_Num,
           Samples_Num):

    Raw_file_list_all = []
    Raw_file_list_all = Get_Raw_Data_File_List(Raw_Data_Path, Training_File_Location_Filters)

    Person_Present_Num_No = 0
    Person_Present_Num_Yes = 0
    Raw_File_Num_Pack = 0

    for file in Raw_file_list_all:
        if com.Get_Person_Present_Status(file) == hd.Person_Present_No:
            Person_Present_Num_No = Person_Present_Num_No + 1
        elif com.Get_Person_Present_Status(file) == hd.Person_Present_Yes:
            Person_Present_Num_Yes = Person_Present_Num_Yes + 1

    Raw_File_Num_Pack = min(Person_Present_Num_No, Person_Present_Num_Yes)
    
    Raw_file_list_No = []
    Raw_file_list_Yes = []
    i = 0
    while i < len(Raw_file_list_all):
        file = Raw_file_list_all[i]
        if len(Raw_file_list_No) == Raw_File_Num_Pack and len(Raw_file_list_Yes) == Raw_File_Num_Pack:
            break
        if com.Get_Person_Present_Status(file) == hd.Person_Present_No and len(Raw_file_list_No) < Raw_File_Num_Pack:
            Raw_file_list_No.append(file)
            del Raw_file_list_all[i]
            i = i - 1
        elif com.Get_Person_Present_Status(file) == hd.Person_Present_Yes and len(Raw_file_list_Yes) < Raw_File_Num_Pack:
            Raw_file_list_Yes.append(file)
            del Raw_file_list_all[i]
            i = i - 1
        i = i + 1

    DataSet_Test_List = []
    Hour_List_Start = []
    Hour_List_End = []
    Raw_File_Test_List_Yes = []
    Raw_File_Test_List_No = []
    Hours_Per_Period = int(24/hd.Time_Period_Num)
    Test_File_Num = int(Raw_File_Num_Pack * 0.1)
    Test_File_Num_Per_Period = int(Test_File_Num/hd.Time_Period_Num)
    for i in range(hd.Time_Period_Num):
        Hour_List_Start.append(i*Hours_Per_Period)
        Hour_List_End.append((i + 1) * Hours_Per_Period - 1)
        DataSet_Test_List.append(Raw2D_Data_Set_Path + Raw2D_Test + "_" + str(i) + hd.Raw_Data_File_Name_Extension)

    for i in range(hd.Time_Period_Num):
        File_List_Temp_Yes = []
        File_List_Temp_No = []
        j = 0
        while j < len(Raw_file_list_Yes):
            Hour_Yes = com.Get_File_Create_Time(Raw_file_list_Yes[j].replace(Raw_Data_Path, Path_Full_Band))
            if Hour_Yes >=Hour_List_Start[i] and Hour_Yes <= Hour_List_End[i] and len(File_List_Temp_Yes) <= Test_File_Num_Per_Period:
                File_List_Temp_Yes.append(Raw_file_list_Yes[j])
                del Raw_file_list_Yes[j]
                j = j - 1
            j = j + 1

        j = 0
        while j < len(Raw_file_list_No):
            Hour_No= com.Get_File_Create_Time(Raw_file_list_No[j].replace(Raw_Data_Path, Path_Full_Band))
            if Hour_No >=Hour_List_Start[i] and Hour_No <= Hour_List_End[i] and len(File_List_Temp_No) <= Test_File_Num_Per_Period:
                File_List_Temp_No.append(Raw_file_list_No[j])
                del Raw_file_list_No[j]
                j = j - 1
            j = j + 1
        Raw_File_Test_List_Yes.append(File_List_Temp_Yes)
        Raw_File_Test_List_No.append(File_List_Temp_No)

    for i in range(hd.Time_Period_Num):
        print("Hour from: " + str(Hour_List_Start[i]) + " to : " + str(Hour_List_End[i]))
        DumpToFile(Raw_File_Test_List_Yes[i] + Raw_File_Test_List_No[i],
                    DataSet_Test_List[i],
                    Reqs_Num,
                    Samples_Num)

    if len(Raw_file_list_Yes) > len(Raw_file_list_No):
        for file in Raw_file_list_all:
            if len(Raw_file_list_No) == len(Raw_file_list_Yes):
                break
            if com.Get_Person_Present_Status(file) == hd.Person_Present_No and len(Raw_file_list_No) < len(Raw_file_list_Yes):
                Raw_file_list_No.append(file)
    else:
        for file in Raw_file_list_all:
            if len(Raw_file_list_Yes) == len(Raw_file_list_No):
                break
            if com.Get_Person_Present_Status(file) == hd.Person_Present_Yes and len(Raw_file_list_Yes) < len(Raw_file_list_No):
                Raw_file_list_Yes.append(file)


    File_Num_Per_Set = int(Raw2D_File_Num_Per_DataSet/2)
    DataSet_File_Num = int(min(len(Raw_file_list_Yes), len(Raw_file_list_No))/ File_Num_Per_Set)


    for i in range(DataSet_File_Num + 1):
        DataSet_Train = Raw2D_Data_Set_Path + Raw2D_Train + "_" + str(i) + hd.Raw_Data_File_Name_Extension
        DataSet_Valid = Raw2D_Data_Set_Path + Raw2D_Valid + "_" + str(i) + hd.Raw_Data_File_Name_Extension

        start = i * File_Num_Per_Set
        end = (i+1) * File_Num_Per_Set
        if i == DataSet_File_Num:
            end = min(len(Raw_file_list_Yes), len(Raw_file_list_No))

        train_file_num = int((end - start) * 0.8)
        valid_file_num = (end - start) - train_file_num
        DumpToFile(Raw_file_list_Yes[start: start + train_file_num] + Raw_file_list_No[start:start + train_file_num],
                   DataSet_Train,
                   Reqs_Num,
                   Samples_Num)

        DumpToFile(Raw_file_list_Yes[start + train_file_num:end] + Raw_file_list_No[start + train_file_num:end],
                  DataSet_Valid,
                   Reqs_Num,
                   Samples_Num)


