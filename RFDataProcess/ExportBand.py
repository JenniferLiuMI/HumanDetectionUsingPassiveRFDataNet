"""
Created by Bing Liu
Export  bands data
"""
import sys
import numpy as np
import Header as hd
import threading
from time import sleep
import datetime
import pickle
from time import sleep
import Common as com

def Get_Data_File_List(Full_Band_Folder):
    files_list = []
    data_files = com.Get_File_name_List(Full_Band_Folder)
    for file in data_files:
        files_list.append(file)
    return files_list


def Export_To_File(Full_Band_Folder, p_raw_file_list, freqs_export, Export_Folder):

    for raw_file in p_raw_file_list:
        f = open(raw_file, "rb")
        freqs = pickle.load(f)
        status = pickle.load(f)
        raw = pickle.load(f)

        sorter = np.argsort(freqs)
        indices = np.reshape(sorter[np.searchsorted(freqs, freqs_export, sorter=sorter)], len(freqs_export))

        status_export = np.take(status, indices, axis = 0)
        raw_export = np.take(raw, indices, axis = 0)
        export_file_name = raw_file.replace(Full_Band_Folder, Export_Folder)

        with open(export_file_name, 'wb') as f:
            pickle.dump(freqs_export, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(status_export, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(raw_export, f, pickle.HIGHEST_PROTOCOL)

def Export(Export_Freq_List_File, Full_Band_Folder, Export_Folder):

    freqs_export = com.Load_Feq_List(Export_Freq_List_File )
    raw_file_list = Get_Data_File_List(Full_Band_Folder)

    end = 0
    start = 0
    threads = []

    files_num = len(raw_file_list)
    step = 50
    thread_max = 30
    threads.clear()

    #Export_To_File(Full_Band_Folder, raw_file_list[0:3], freqs_export, Export_Folder)

    while end < files_num:
        if len(threads) < thread_max:
            end = start + step
            if end > files_num:
                end = files_num
            if(end > start):
                thread_pack_IQ_data = threading.Thread(target=Export_To_File, args=(Full_Band_Folder, raw_file_list[start:end], freqs_export, Export_Folder))
                threads.append(thread_pack_IQ_data)
                thread_pack_IQ_data.start()
                print(thread_pack_IQ_data.name + " " + str(start) + " " + str(end))
                sleep(0.1)
                start = end
        else:
            for thread in threads:
                thread.join()
            threads.clear()

