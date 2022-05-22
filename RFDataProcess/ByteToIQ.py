"""
Created by Bing Liu
Conver raw data to IQ data
"""
import sys
sys.path.append('../RFDataCollect')
import numpy as np
import Header as hd
import threading
from time import sleep
import datetime
import pickle
from time import sleep
import Common as com

#Get raw data file list
def Get_Convert_Raw_Data_File_List(Raw_Data_Path):
    files_list = []
    data_files = com.Get_File_name_List(Raw_Data_Path)
    for file in data_files:
        files_list.append(file)
    return files_list

#Conver raw data to IQ data
def Raw_To_IQ_File(p_raw_file_list):

    raw_file_list = []
    for file in p_raw_file_list:
        raw_file_list.append(file)

    file_list_len = len(raw_file_list)
    feq_list = []
    status_list = []
    raw_data_bytes_list = []
    IQData_list = []
    
    for raw_file in raw_file_list:
        f = open(raw_file, "rb")
        feq_arr = pickle.load(f)
        status_arr = pickle.load(f)
        raw_data_bytes_arr = pickle.load(f)

        feq_list.append(feq_arr)
        status_list.append(status_arr)
        raw_data_bytes_list.append(raw_data_bytes_arr)
        f.close()

    for raw_data_bytes_arr in raw_data_bytes_list:
        IQData_arr = []
        IQData_arr.clear()
        for raw_data_bytes in raw_data_bytes_arr:

            data = np.ctypeslib.as_array(raw_data_bytes)
            iq = data.astype(np.float32).view(np.complex64)
            iq /= 127.5
            iq -= (1 + 1j)
            iq_real = np.stack((iq.real,iq.imag), axis = 1)
            IQData_arr.append(iq_real)
        IQData_list.append(np.copy(IQData_arr))

    for i in range(file_list_len):
        raw_file_name = raw_file_list[i]
        IQ_file_name = raw_file_name.replace(hd.Raw_Data_File_Name_Prefix, hd.IQ_Data_File_Name_Prefix)
        IQ_file_name = IQ_file_name.replace(hd.Raw_Data_File_Path, hd.IQ_Data_File_Path)
        with open(IQ_file_name, 'wb') as f:
            pickle.dump(feq_list[i], f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(status_list[i], f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(IQData_list[i], f, pickle.HIGHEST_PROTOCOL)
    #print("Duemp done." + datetime.datetime.now().strftime("%H:%M:%S"))   
    return

if __name__ == '__main__':
    raw_file_list = []
    end = 0
    start = 0
    threads = []
    #Uncomment below code to convert
    #Done raw_file_list = Get_Convert_Raw_Data_File_List(hd.Raw_Data_Active_Band)
    #Done raw_file_list = Get_Convert_Raw_Data_File_List(hd.Raw_Data_Active_Band_ExcludedCell)
    #Done raw_file_list = Get_Convert_Raw_Data_File_List(hd.Raw_Data_Inactive_Band)
    #Done raw_file_list = Get_Convert_Raw_Data_File_List(hd.Raw_Data_Inactive_Band_ExcludedCell)
    #Done raw_file_list = Get_Convert_Raw_Data_File_List(hd.Raw_Data_Random_Band)
    raw_file_list = Get_Convert_Raw_Data_File_List(hd.Raw_Data_Selective_Band)

    files_num = len(raw_file_list)
    step = 50
    thread_max = 1
    threads.clear()

    Raw_To_IQ_File(raw_file_list[:2])
    raw_file_list = raw_file_list[31050:]
    
    while end < files_num:
        if len(threads) < thread_max:
            end = start + step
            if end > files_num:
                end = files_num
            if(end > start):
                thread_pack_IQ_data = threading.Thread(target=Raw_To_IQ_File, args=(raw_file_list[start:end],))
                threads.append(thread_pack_IQ_data)
                thread_pack_IQ_data.start()
                print(thread_pack_IQ_data.name + " " + str(start) + " " + str(end))
                sleep(0.1)
                start = end
        else:
            for thread in threads:
                thread.join()
            threads.clear()