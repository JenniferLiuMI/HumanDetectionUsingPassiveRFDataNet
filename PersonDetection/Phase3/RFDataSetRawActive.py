"""
Created by Bing Liu
Functions to process raw data
"""

import pickle
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
sys.path.insert(1, '/media/win/Code/RFResearch/PersonDetection')
#sys.path.insert(1, 'E:\Code\RFResearch\PersonDetection')
import Header as hd
import Common as com
sys.path.insert(1, '/media/win/Code/RFResearch/PersonDetection/Phase1')
#sys.path.insert(1, 'E:\Code\RFResearch\PersonDetection\Phase1')
import Raw2DCreateDataSet
import Com_Power

def create(Select_Band_Num, Samples_Num):
    Raw_Data_File_Path = hd.Raw_Data_File_Path
    Data_Set_Path = hd.GAN_Raw2D_Data_Set_Path_Band.format(Select_Band_Num, Samples_Num)

    if os.path.exists(Data_Set_Path) == False:
        os.mkdir(Data_Set_Path)

    Create_Raw(Raw_Data_File_Path,
                                Data_Set_Path,
                                hd.Raw2D_Data_Set_Train,
                                hd.Raw2D_Data_Set_Valid,
                                hd.Raw2D_Data_Set_Test,
                                400,
                                ["01"],
                                Select_Band_Num,
                                Samples_Num)


def Create_Raw(Raw_Data_Path, 
           Raw2D_Data_Set_Path, 
           Raw2D_Train, 
           Raw2D_Valid, 
           Raw2D_Test, 
           Raw2D_File_Num_Per_DataSet, 
           Training_File_Location_Filters,
           Select_Band_Num,
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
    Data_Set_Test_File_Name = Raw2D_Data_Set_Path + Raw2D_Test + hd.Raw_Data_File_Name_Extension
    DumpToFile(Raw_file_list_Yes[0:test_file_num] + Raw_file_list_No[0:test_file_num],
                Data_Set_Test_File_Name,
                Select_Band_Num,
                Samples_Num)

    test_file_num_rest = Raw_File_Num_Pack - test_file_num
    File_Num_Per_Set = int(Raw2D_File_Num_Per_DataSet/2)
    DataSet_File_Num = int((test_file_num_rest) / File_Num_Per_Set)

    Raw_file_list_Yes_Rest = []
    Raw_file_list_No_Rest = []

    Raw_file_list_Yes_Rest = Raw_file_list_Yes[test_file_num:Raw_File_Num_Pack]
    Raw_file_list_No_Rest = Raw_file_list_No[test_file_num:Raw_File_Num_Pack]

    for i in range(DataSet_File_Num + 1):
        Data_Set_Train_File_Name = Raw2D_Data_Set_Path + Raw2D_Train + "_" + str(i) + hd.Raw_Data_File_Name_Extension
        Data_Set_Valid_File_Name = Raw2D_Data_Set_Path + Raw2D_Valid + "_" + str(i) + hd.Raw_Data_File_Name_Extension

        start = i * File_Num_Per_Set
        end = (i+1) * File_Num_Per_Set
        if i == DataSet_File_Num:
            end = len(Raw_file_list_Yes_Rest)

        if end - start < File_Num_Per_Set and i >0:
            break

        train_file_num = int((end - start) * 0.8)
        valid_file_num = (end - start) - train_file_num
        DumpToFile(Raw_file_list_Yes_Rest[start: start + train_file_num] + Raw_file_list_No_Rest[start:start + train_file_num],
                   Data_Set_Train_File_Name,
                   Select_Band_Num,
                   Samples_Num)

        DumpToFile(Raw_file_list_Yes_Rest[start + train_file_num:end] + Raw_file_list_No_Rest[start + train_file_num:end],
                  Data_Set_Valid_File_Name,
                   Select_Band_Num,
                   Samples_Num)


def CreateDataSet_Column(Raw_file_list, Select_Band_Num, Samples_Num):

    Freqs_List_Full = com.Load_Feq_List(hd.Full_Band_Freq_List_File )

    Freqs_Selected = np.array(com.Import_From_Text_To_Int(hd.PCA_Selected_Band_List_File.format(Select_Band_Num)))
    Freqs_Selected = np.sort(Freqs_Selected)
    Seleced_Freq_Index = np.in1d(Freqs_List_Full, Freqs_Selected).nonzero()[0]  

    Lables = []
    Raws_All = []
    for raw_file in Raw_file_list:
        print(raw_file + "  " + str(com.Get_File_Create_Time(raw_file.replace(hd.Raw_Data_File_Path_Active_Band, hd.Raw_Data_Full_Band))))

        f = open(raw_file, "rb")
        freq_arr = pickle.load(f)
        lable_arr = pickle.load(f)
        Raw_arr = pickle.load(f)
        
        Raw_arr_seleted = np.take(Raw_arr, Seleced_Freq_Index, axis=0)
        sample_step = int(np.shape(Raw_arr)[1]/Samples_Num)

        Raw_data_arr_temp=(np.average(Raw_arr_seleted.reshape((Raw_arr_seleted.shape[0], Samples_Num, -1) ),axis=2)).reshape(Select_Band_Num, Samples_Num, 1)
        
        Lables.append(lable_arr[0])
        Raws_All.append(Raw_data_arr_temp)

    return np.asarray(Lables, dtype=hd.Raw_Data_Type_Str),  np.asarray(Raws_All, dtype=hd.Raw_Data_Type_Str)

def DumpToFile(Raw_file_list, Raw_Data_Set_File_Name, Select_Band_Num, Samples_Num):
    print("Dump " + str(len(Raw_file_list)) + " files to : " + Raw_Data_Set_File_Name)
    Lables, Raws = CreateDataSet_Column(shuffle(Raw_file_list), Select_Band_Num, Samples_Num)
    with open(Raw_Data_Set_File_Name, 'wb') as f:
        pickle.dump((Lables, Raws), f, pickle.HIGHEST_PROTOCOL)
   
def load(Select_Band_Num, Samples_Num, figure_dir):
    Freqs_Selected = np.array(com.Import_From_Text_To_Int(hd.PCA_Selected_Band_List_File.format(Select_Band_Num)))
    Freqs_Selected = np.sort(Freqs_Selected)
    Data_Set_Path = hd.GAN_Raw2D_Data_Set_Path_Band.format(Select_Band_Num, Samples_Num)

    train_file_list = []
    train_file_list = com.Get_File_List(Data_Set_Path, hd.Raw2D_Data_Set_Train);

    valid_file_list = []
    valid_file_list = com.Get_File_List(Data_Set_Path, hd.Raw2D_Data_Set_Valid);

    n_train_file = len(train_file_list)
    
    f_Train = open(train_file_list[0], "rb")
    train_Lables, train_Raws = pickle.load(f_Train)

    f_Valid = open(valid_file_list[0], "rb")
    valid_Lables, valid_Raws = pickle.load(f_Valid)

    x_train_all = train_Raws
    y_train_all = train_Lables

    x_valid_all = valid_Raws
    y_valid_all = valid_Lables

    figure_name = os.path.join(figure_dir, "Train_Power" + hd.JPG_File_Extension)
    Com_Power.Draw_Freqs_Powers_Avg_Train(figure_name, Freqs_Selected, x_train_all, y_train_all, x_valid_all, y_valid_all)

    for j in range(1, n_train_file):
        f_Train = open(train_file_list[j], "rb")
        train_Lables, train_Raws = pickle.load(f_Train)

        f_Valid = open(valid_file_list[j], "rb")
        valid_Lables, valid_Raws = pickle.load(f_Valid)

        x_train_all = np.concatenate((x_train_all, train_Raws), axis=0)
        y_train_all = np.concatenate((y_train_all, train_Lables), axis=0)
        
        x_valid_all = np.concatenate((x_valid_all, valid_Raws), axis=0)
        y_valid_all = np.concatenate((y_valid_all, valid_Lables), axis=0)

    Com_Power.Draw_Freqs_Powers_Avg_Train(figure_name, Freqs_Selected, x_train_all, y_train_all, x_valid_all, y_valid_all)
    return Freqs_Selected, x_train_all, y_train_all, x_valid_all, y_valid_all
#create(2, 4800)
create(8, 4800)
#load(800, 300, './media/win/Code/RFResearch/GANout/Raw800/figures/')