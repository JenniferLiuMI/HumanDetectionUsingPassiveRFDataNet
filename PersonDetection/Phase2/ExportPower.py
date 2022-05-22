"""
Created by Bing Liu
Caculate and export powers in frequency bands
"""

from sklearn.utils import shuffle
import numpy as np
import os
import pickle
import time
import math
import Header as hd

#Caculate the powers in frequency bands
def Get_Freqs_Powers(raw_file_list):

    power_list_all = []
    for raw_file in raw_file_list:
        f = open(raw_file, "rb")
        freqs = pickle.load(f)
        raw_data_per_file = np.asarray(pickle.load(f))
        raw_data_per_file = raw_data_per_file/127.5 - 1
        power_per_file = np.linalg.norm(raw_data_per_file, axis=1)
        power_per_file = power_per_file*power_per_file
        db = 10*np.log10(power_per_file/len(raw_data_per_file[0])/2)
        power_list_all.append(db)
    power_arry = np.asarray(power_list_all)
    return power_arry

#Get the file name list
def Get_File_name_List(path, location, status, num = -1):

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
        if hour>6 and hour< 22:
            file_list_filtered.append(file)
            count = count + 1
    return file_list_filtered

#Export the caculated powers
def Export_Power(band, path):

    file_name_list_01_Yes= []
    file_name_list_01_No= []
    file_name_list_05_Yes= []
    file_name_list_05_No= []

    band = band + "_File_"
    home = "01"
    office = "05"
    yes = "_Yes"
    no = "_No"

    record_len = 100
    file_name_list_01_Yes = Get_File_name_List(path, home, str(hd.Person_Present_Yes), record_len)
    file_name_list_01_No = Get_File_name_List(path, home, str(hd.Person_Present_No), record_len)
    file_name_list_05_Yes = Get_File_name_List(path, office, str(hd.Person_Present_Yes), record_len)
    file_name_list_05_No = Get_File_name_List(path, office, str(hd.Person_Present_No), record_len)

    export_power_file_name = hd.Phase2_Root_Path + band + str(record_len) + '_' + home + yes + hd.Raw_Data_File_Name_Extension
    with open(export_power_file_name, 'wb') as f_export:
        pickle.dump((Get_Freqs_Powers(file_name_list_01_Yes)), f_export, pickle.HIGHEST_PROTOCOL)

    export_power_file_name = hd.Phase2_Root_Path + band + str(record_len) + '_' + home + no + hd.Raw_Data_File_Name_Extension
    with open(export_power_file_name, 'wb') as f_export:
        pickle.dump((Get_Freqs_Powers(file_name_list_01_No)), f_export, pickle.HIGHEST_PROTOCOL)

    export_power_file_name = hd.Phase2_Root_Path + band + str(record_len) + '_' + office + yes + hd.Raw_Data_File_Name_Extension
    with open(export_power_file_name, 'wb') as f_export:
        pickle.dump((Get_Freqs_Powers(file_name_list_05_Yes)), f_export, pickle.HIGHEST_PROTOCOL)
    
    export_power_file_name = hd.Phase2_Root_Path + band + str(record_len) + '_' + office + no + hd.Raw_Data_File_Name_Extension
    with open(export_power_file_name, 'wb') as f_export:
        pickle.dump((Get_Freqs_Powers(file_name_list_05_No)), f_export, pickle.HIGHEST_PROTOCOL)

#Get the file generation time
def Get_File_Create_Time(file):
    year,month,day,hour,minute,second=time.localtime(os.path.getmtime(file))[:-3]
    return hour

#Export powers
if __name__ == '__main__':
    Export_Power("Selective", hd.Raw_Data_Selective_Band )
    Export_Power("Active", hd.Raw_Data_Active_Band )
    Export_Power("ActiveExcludedCell", hd.Raw_Data_Active_Band_ExcludedCell )
    Export_Power("Random", hd.Raw_Data_Random_Band )
