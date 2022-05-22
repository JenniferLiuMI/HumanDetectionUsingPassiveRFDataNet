"""
Created by Bing Liu
Common funtions
"""

import os
import time
import os.path
import numpy as np
import Header as hd
import csv
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import datetime
import logging
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from pathlib import Path

#Get label name
def Get_Label_Name(label):
    if label == hd.Person_Present_No:
        return 'Occupied'
    else:
        return 'Unoccupied'

#Import text and convert to int
def Import_From_Text_To_Int(file_name):
    with open(file_name, "r") as f:
        list = []
        for item in f:
            list.append(int(item.replace("\n","")))
    return list

#Convert to one hot
def To_One_Hot(y):
    y = y.astype(np.int)
    y_temp = np.zeros((len(y), 2), dtype=np.int)
    for i in range(len(y)):
        y_temp[i, y[i]] = 1
    return y_temp

#Get raw data file list by location
def Get_Raw_Data_File_List_By_Location(Raw_Data_File_Path, Training_File_Location_Filters):
    files_list = []
    data_files = Get_File_name_List(Raw_Data_File_Path)
    for file in data_files:
       location_code = Get_Location_Code(file, Raw_Data_File_Path)
       for code in Training_File_Location_Filters:
           if code == location_code: 
                files_list.append(file)
                break
    return shuffle(files_list)

#Get file list by location
def Get_File_List_By_Location(File_Path, File_Extention, Location_Filters):
    files_list = []
    data_files = Get_File_name_List(File_Path)
    for file in data_files:
       location_code = Get_Location_Code_1(file, File_Extention)
       for code in Location_Filters:
           if code == location_code: 
                files_list.append(file)
                break
    return files_list

#Get file list by time
def Get_File_List_By_Hour(files_list, From, To):
    files_list_temp = []
    for file in files_list:
       create_hour = Get_File_Create_Time(file)
       if create_hour>= From and create_hour<=To:
            files_list_temp.append(file)
    return files_list_temp

#Get training IQ file list
def Get_IQ_Train_File_List():
    files_list = []
    data_files = Get_File_name_List(hd.IQ_Train_Path)
    for file in data_files:
        if file[len(file)-len(hd.IQ_Train_Path):-1].find(hd.IQ_Train) > -1:
            files_list.append(file)
    return files_list

#Get validation IQ file list
def Get_IQ_Valid_File_List():
    files_list = []
    data_files = Get_File_name_List(hd.IQ_Train_Path)
    for file in data_files:
        if file[len(file)-len(hd.IQ_Train_Path):-1].find(hd.IQ_Valid) > -1:
            files_list.append(file)
    return files_list

#Get test IQ file list
def Get_IQ_Test_File_List():
    files_list = []
    data_files = com.Get_File_name_List(hd.IQ_Train_Path)
    for file in data_files:
        if file[len(file)-len(hd.IQ_Train_Path):-1].find(hd.IQ_Test) > -1:
            files_list.append(file)
    return files_list

#Get training raw file list
def Get_Raw_Train_File_List():
    files_list = []
    data_files = Get_File_name_List(hd.Raw_Train_Path)
    for file in data_files:
        if file[len(file)-len(hd.Raw_Train_Path):-1].find(hd.Raw_Train) > -1:
            files_list.append(file)
    return files_list

#Get validation raw file list
def Get_Raw_Valid_File_List():
    files_list = []
    data_files = Get_File_name_List(hd.Raw_Train_Path)
    for file in data_files:
        if file[len(file)-len(hd.Raw_Train_Path):-1].find(hd.Raw_Valid) > -1:
            files_list.append(file)
    return files_list

#Get test raw file list
def Get_Raw_Test_File_List():
    files_list = []
    data_files = com.Get_File_name_List(hd.Raw_Train_Path)
    for file in data_files:
        if file[len(file)-len(hd.Raw_Train_Path):-1].find(hd.Raw_Test) > -1:
            files_list.append(file)
    return files_list

#Get file list
def Get_File_List(Data_Set_Path, File_Type):
    files_list = []
    data_files = Get_File_name_List(Data_Set_Path)
    for file in data_files:
        if Path(file).name.find(File_Type) > -1:
            files_list.append(file)
    return files_list

#Get person present status
def Get_Person_Present_Status(file_name):
    index_S = len(file_name) - len(hd.Raw_Data_File_Name_Extension) - 1
    Person_Present_Status = int(file_name[index_S: index_S+1])
    return Person_Present_Status

#Load dataset
def LoadDataSet(train_file):
    IQDataSet = hd.IQ_Train_Path + hd.IQ_Train + "_" + str(0) + hd.Raw_Data_File_Name_Extension
    f = open(IQDataSet, "rb")
    XYZ_Traing, XYZ_Valid, XYZ_Test = pickle.load(f)

#Caculate samples numbers
def Cal_Samples_Num(Sample_Rate, Milliseconds):
    return int(Sample_Rate * (Milliseconds/1000))

#Get the last raw data file name
def Get_Last_Raw_Data_File_name():
    max_num = Get_Raw_Data_File_Index_Max()
    raw_data_file_list = Get_File_name_List(hd.Raw_Data_File_Path)
    if len(raw_data_file_list) > 0:
        for file_name in raw_data_file_list:
            if file_name.find(max_num) > 0:
                return file_name

#Generate the raw data file name
def Generate_Raw_Data_File_name(location_code, person_status):
    file_name = hd.Raw_Data_File_Name_Prefix + str(int(Get_Raw_Data_File_Index_Max()) + 1).zfill(len(hd.Raw_Data_File_Name_Start_Index)) + '_' + location_code +'_' + str(person_status) + hd.Raw_Data_File_Name_Extension
    return hd.Raw_Data_File_Path +  file_name

#Get the raw data file name maximun index number
def Get_Raw_Data_File_Index_Max():
    raw_data_file_list = Get_File_name_List(hd.Raw_Data_File_Path)
    file_index_arry = []
    if len(raw_data_file_list) > 0:
        for file_name in raw_data_file_list:
            file_index = str(Get_Raw_Data_File_Index(file_name))
            file_index_arry.append(file_index)
        max_index = str(max(file_index_arry))
        max_index = max_index[0:(len(max_index))]
        return str(max_index)
    else:
        return hd.Raw_Data_File_Name_Start_Index

#Get the raw data file name index
def Get_Raw_Data_File_Index(file_name):
    index_S = len(hd.Raw_Data_File_Path) + len(hd.Raw_Data_File_Name_Prefix)
    index_E = len(hd.Colect_Location_Code_StartIndex) + len(hd.Raw_Data_File_Name_Extension) + 3
    index = file_name[index_S: -index_E] 
    return index

#Get location code
def Get_Location_Code_1(file_name, file_extention):
    len_file_extention= len(file_extention)
    location_code = file_name[-(len_file_extention+len(hd.Colect_Location_Code_StartIndex)+len(hd.Colect_Status_Code_StartIndex)): -(len_file_extention+len(hd.Colect_Status_Code_StartIndex))] 
    return location_code

#Get location code
def Get_Location_Code(file_name, Raw_Data_File_Path):
    index_S = len(Raw_Data_File_Path) + len(hd.Raw_Data_File_Name_Prefix) + len(hd.Raw_Data_File_Name_Start_Index) + 1
    len_location_code = len(hd.Colect_Location_Code_StartIndex)
    location_code = file_name[index_S: index_S + len_location_code] 
    return location_code

def Get_Location_Code_IQ_File(file_name):
    index_S = len(hd.IQ_Data_File_Path) + len(hd.IQ_Data_File_Name_Prefix) + len(hd.Raw_Data_File_Name_Start_Index) + 1
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

#Rename raw data files
def Rename_Raw_Data_Files():
    #raise Exception('Do not run: Rename_Raw_Data_Files()')
    raw_data_file_list = Get_File_name_List("C:/RFRoot/RFRawDataNewy/")
    file_index_arry = []
    if len(raw_data_file_list) > 0:
        for file_name in raw_data_file_list:
            dst = file_name.replace("_02_", "_01_")
            os.rename(file_name, dst)

#Setup log file
def Log_Setup_Train(folder):
    log_file_name = folder + "/Log/Train" + hd.Log_File_Extension
    logging.basicConfig(filename=log_file_name,level=logging.DEBUG,format= '%(message)s')

def Log_Setup(folder, file_name):
    log_file_name = folder + file_name + hd.Log_File_Extension
    logging.basicConfig(filename=log_file_name,level=logging.DEBUG,format= '%(message)s')

#Log status change to file
def Log_Status_Change(status_change_arr):
    for status in status_change_arr:
        logging.info('Status: ' + status)

#Log information
def Log_Info_Arr(msg_arr):
    for msg in msg_arr:
        logging.info(msg)

#Write to log file
def Log_Info(msg):
    logging.info(msg)

#Write to log file
def Log_Inf1o():
    logging.info("info message")
    logging.warn("warn message")
    logging.error("error message")
    logging.critical("critical message")

#Write record to master file
def Write_Master_File(data_file_name, location, time_stemp ):
    rows = []
    rows.append([data_file_name,location,str(time_stemp)])

    if (os.path.exists(master_file_name) == False):

        header = [hd.Master_File_Header_File_Name, \
                        hd.Master_File_Header_Location, \
                        hd.Master_File_Header_Time_Stemp]
        with open(master_file_name, 'wt', newline ='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
    else:
        with open(master_file_name, 'a', newline ='') as file:
            writer = csv.writer(file, delimiter=',')
            for row in rows:
                writer.writerow(row)

#Convert status to number
def Conver_Status(serial_status):
    if serial_status.find(str(hd.Person_Present_No)) > -1:
        return int(hd.Person_Present_No)
    elif serial_status.find(str(hd.Person_Present_Yes)) > -1:
        return int(hd.Person_Present_Yes)
    else:
        return -1

#Add record to raw data file
def Save_To_Raw_Data_file(file_name, SDR_bytes):
    with open(file_name, "ab+") as f:
        f.write(SDR_bytes)

#Read the raw data back from file
def Read_From_Raw_Data_file(file_name):
    with open(file_name, "rb") as f:
        bytes_readback = []
        readback = f.read()
        for byte in readback:
            bytes_readback.append(byte)
    return bytes_readback

#Pack raw data    
def Pack_Raw_Data(feq_arr, status_arr,raw_data_arr):
    x, y,z = [], [],[]
    count = 0
    for feq in feq_arr:
        x.append(feq)
        y.append(status_arr[count])
        z.append(raw_data_arr[count])
        count +=1
    return np.asarray(x, dtype='uint32'), np.asarray(y, dtype='uint8'), np.asarray(z, dtype='uint8')

#Dump raw data to file
def Dump_Raw_Data_To_File(raw_data_file_name, feq_arr, status_arr,raw_data_arr):
    feq, status, raw_data = Pack_Raw_Data(feq_arr, status_arr, raw_data_arr)
    with open(raw_data_file_name, 'wb') as f:
        pickle.dump(feq, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(status, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(raw_data, f, pickle.HIGHEST_PROTOCOL)

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

#Get frequency list
def Get_Feq_List(FileName):
    freqs = []
    with open(FileName) as f:
        for freq in f:
            freqs.append(freq)
    return freqs

#Get frequency list
def Load_Feq_List(FreqFile):
    freq_arr = []
    with open(FreqFile, 'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            freq_arr.append(inner_list)
    return (np.asarray(freq_arr)).astype(int).reshape(len(freq_arr))

#Get file create time
def Get_File_Create_Time(file):
    if hd.Is_Windows == True:
        year,month,day,hour,minute,second=time.localtime(os.path.getmtime(file))[:-3]
        return hour
    else:
        stat = os.stat(file)
        created = os.path.getctime(file)
        year,month,day,hour,minute,second=time.localtime(created)[:-3]
        return hour
