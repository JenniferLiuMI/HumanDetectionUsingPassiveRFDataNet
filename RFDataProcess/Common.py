"""
Created by Bing Liu
Common functions used during data processing
"""
import os
import os.path
import numpy as np
import Header as hd
import csv
import datetime
import logging
import pickle
import re
from sklearn.utils import shuffle
import time

#Get file name list
def Get_File_name_List_1(path, location, status, num = -1):

    all = os.listdir(path)
    file_list_all = [x for x in all if re.search('_'+location+'_' + status + '.pkl', x)]
    file_list_all = shuffle(file_list_all)
    file_list_filtered = []
    count = 0
    if num == -1:
       num = len(file_list_all)
    index = 0
    while count < num:
        file = path + file_list_all[index]
        index = index + 1
        hour = Get_File_Create_Time(file)
        if hour>6 and hour< 23:
            file_list_filtered.append(file)
            count = count + 1
    return file_list_filtered

#Get file name list
def Get_File_name_List_2(path, location, status, num = -1):

    all = os.listdir(path)
    file_list_all = [x for x in all if re.search('_'+location+'_' + status + '.pkl', x)]
    file_list_all = shuffle(file_list_all)
    file_list_filtered = []
    count = 0
    if num == -1:
       num = len(file_list_all)
    index = 0
    while count < num:
        file = path + file_list_all[index]
        index = index + 1
        file_list_filtered.append(file)
        count = count + 1
    return file_list_filtered

#Get the raw data file name index
def Get_Raw_Data_File_Index(file_name):
    index_S = len(hd.Raw_Data_File_Path) + len(hd.Raw_Data_File_Name_Prefix)
    index_E = len(hd.Colect_Location_Code_StartIndex) + len(hd.Raw_Data_File_Name_Extension) + 3
    index = file_name[index_S: -index_E] 
    return index

#Get file names
def Get_Location_Code(file_name):
    index_S = len(hd.Raw_Data_File_Path) + len(hd.Raw_Data_File_Name_Prefix) + len(hd.Raw_Data_File_Name_Start_Index) + 1
    len_location_code = len(hd.Colect_Location_Code_StartIndex)
    location_code = file_name[index_S: index_S + len_location_code] 
    return location_code

#Get file names
def Get_File_name_List(path):
    file_name_list= []
    for file_name in os.listdir(path):
        if(file_name.endswith(hd.Raw_Data_File_Name_Extension)):
            file_name_list.append(path + file_name)
    return file_name_list

#Read the raw data back from file
def Read_From_Raw_Data_file(file_name):
    with open(file_name, "rb") as f:
        bytes_readback = []
        readback = f.read()
        for byte in readback:
            bytes_readback.append(byte)
    return bytes_readback

#Load raw data from file
def Load_Raw_Data(raw_data_file_name):
    with open(raw_data_file_name, 'rb') as f:
        feq = pickle.load(f)
        status = pickle.load(f)
        raw_data = pickle.load(f)
        return feq,status,raw_data

#Load IQ data
def Load_IQ_Data(IQ_data_file_name):
    with open(IQ_data_file_name, 'rb') as f:
        feq = pickle.load(f)
        status = pickle.load(f)
        IQ_data = pickle.load(f)
        return feq,status,IQ_data

#Get status change from serial
def Serial_Get_Status(ser):
    status = ser.read(size=1)
    status = str(status).replace('b', '')
    return(Conver_Status(status))

#Import 
def Import_From_Text(file_name):
    with open(file_name, "r") as f:
        list = []
        for item in f:
            list.append(float(item.replace("\n","")))
    return list

def Import_From_Text_To_Int(file_name):
    with open(file_name, "r") as f:
        list = []
        for item in f:
            list.append(int(item.replace("\n","")))
    return list


def Get_Last_Raw_Data_File_name_Ful():
    max_num = Get_Raw_Data_File_Index_Max_Ful()
    raw_data_file_list = Get_File_name_List(hd.Raw_Data_File_Path_Ful)
    if len(raw_data_file_list) > 0:
        for file_name in raw_data_file_list:
            if file_name.find(max_num) > 0:
                return file_name

def Convert_Bytes_To_IQ( bytes):
    data = np.ctypeslib.as_array(bytes)
    iq = data.astype(np.float32).view(np.complex64)
    iq /= 127.5
    iq -= (1 + 1j)
    return iq

def Convert_Bytes_To_IQ_1( bytes):
    data = np.ctypeslib.as_array(bytes)
    #use normal list
    iq = [complex(i/(255/2) - 1, q/(255/2) - 1) for i, q in zip(bytes[::2], bytes[1::2])]
    return iq

def Generate_Feq_List_Ful():
    feq_arr = []
    feq_arr += Cal_Feq_List(24e6, 1.2e6, 1760e6) 
    return feq_arr

def Remove_Data_File(path, extension):
    file_name_list= []
    for file_name in os.listdir(path):
        if(file_name.endswith(hd.Raw_Data_File_Name_Extension)):
            file_delete = path + file_name
            os.remove(file_delete)
            print("Deleted " + file_delete)