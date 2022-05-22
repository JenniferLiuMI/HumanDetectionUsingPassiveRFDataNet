"""
Created by Bing Liu
Common function used in data collection
"""
import os
import os.path
import numpy as np
import SDRHeader as hd
import csv
import datetime
import logging
import pickle
import serial

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

#Rename raw data files
def Rename_Raw_Data_Files():
    raise Exception('Do not run: Rename_Raw_Data_Files()')
    raw_data_file_list = Get_File_name_List(hd.Raw_Data_File_Path)
    file_index_arry = []
    if len(raw_data_file_list) > 0:
        for file_name in raw_data_file_list:
            dst = file_name[0:26] + '00' + file_name[26:]
            os.rename(file_name, dst)

#Setup log file
def Log_Setup():
    log_file_name = hd.Raw_Data_Log_File_Path + str(datetime.datetime.now().strftime('%Y_%m_%d')) + hd.Log_File_Extension
    #logging.basicConfig(filename=log_file_name,level=logging.DEBUG,format='%(asctime)s %(message)s')
    logging.basicConfig(filename=log_file_name,level=logging.DEBUG,format= '%(message)s')

#Log status change to file
def Log_Status_Change(status_change_arr):
    for status in status_change_arr:
        logging.info('Status: ' + status)

#Log information
def Log_Info(msg_arr):
    for msg in msg_arr:
        logging.info(msg)

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

#Open serial
def Serial_Open():
    ser = serial.Serial(timeout=0)
    ser.baudrate = 9600
    ser.port = 'COM8'
    ser.open()
    return ser

#Get status change from serial
def Serial_Get_Status(ser):
    status = ser.read(size=1)
    status = str(status).replace('b', '')
    return(Conver_Status(status))

#Close serail
def Serial_Close(ser):
    ser.close()

#Generate the even frequency bands list
def Cal_Feq_List(feq_start, step, feq_end):
    feq_arr_temp = []
    num = int((feq_end-feq_start)/step)
    i = 0;
    while(i <= num):
        feq_arr_temp.append(feq_start + i*step)
        i += 1
    return feq_arr_temp

#Generate the selective frequency bands list
def Generate_Feq_List():

    #[R82XX] PLL not locked!
    #0.5MHz - 1.8MHz
    #feq_arr += Cal_Feq_List(500.e3, 10e3, 1.8e6 )
    #1.9MHz - 87MHz
    #feq_arr += Cal_Feq_List(3.0e6, 10e6, 87e6 )        

    feq_arr = []

    #Start from 24MHz, 24MHz - 53MHz
    feq_arr += Cal_Feq_List(24e6, 5e6, 53e6 ) 

    #54MHz - 87MHz
    feq_arr += Cal_Feq_List(54e6, 1e6, 87e6 ) 


    #88MHz - 108MHz
    #feq_arr += Cal_Feq_List(88e6, 1e6, 108e6 )        

    feq_arr.append(88.3e6)
    feq_arr.append(88.7e6)
    feq_arr.append(89.5e6)
    feq_arr.append(89.9e6)
    feq_arr.append(92.3e6)
    feq_arr.append(92.7e6)
    feq_arr.append(93.1e6)
    feq_arr.append(93.9e6)
    feq_arr.append(94.1e6)
    feq_arr.append(94.7e6)
    feq_arr.append(95.5e6)
    feq_arr.append(96.3e6)
    feq_arr.append(97.1e6)
    feq_arr.append(97.9e6)
    feq_arr.append(98.7e6)
    feq_arr.append(99.5e6)
    feq_arr.append(101.1e6)
    feq_arr.append(101.9e6)
    feq_arr.append(102.1e6)
    feq_arr.append(102.7e6)
    feq_arr.append(103.1e6)
    feq_arr.append(103.5e6)
    feq_arr.append(104.3e6)
    feq_arr.append(105.1e6)
    feq_arr.append(105.9e6)
    feq_arr.append(106.7e6)
    feq_arr.append(107.5e6)

    #109MHz - 173MHz
    feq_arr += Cal_Feq_List(109e6, 10e6, 173e6 )        

    #174MHz - 216MHz
    feq_arr += Cal_Feq_List(174e6, 1e6, 216e6 )        

    #217MHz - 469MHz
    feq_arr += Cal_Feq_List(217e6, 10e6, 469e6 )        

    #470MHz - 806MHz
    feq_arr += Cal_Feq_List(470e6, 10e6, 806e6 )        

    #807MHz - 1760MHz
    feq_arr += Cal_Feq_List(807e6, 50e6, 1760e6)        

    return feq_arr