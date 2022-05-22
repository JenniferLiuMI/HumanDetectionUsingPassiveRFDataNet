"""
Created by Bing Liu
The common functions used in phase 2
"""
import os
import os.path
import numpy as np
import Header as hd
import re
from sklearn.utils import shuffle
import time

# Get the last raw data file name
def Get_Last_Raw_Data_File_name():
    max_num = Get_Raw_Data_File_Index_Max()
    raw_data_file_list = Get_File_name_List(hd.Raw_Data_File_Path)
    if len(raw_data_file_list) > 0:
        for file_name in raw_data_file_list:
            if file_name.find(max_num) > 0:
                return file_name

#Get file creation time
def Get_File_Create_Time(file):
    year,month,day,hour,minute,second=time.localtime(os.path.getmtime(file))[:-3]
    return hour

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

# Get the raw data file name index
def Get_Raw_Data_File_Index(file_name):
    index_S = len(hd.Raw_Data_File_Path) + len(hd.Raw_Data_File_Name_Prefix)
    index_E = len(hd.Colect_Location_Code_StartIndex) + len(hd.Raw_Data_File_Name_Extension) + 3
    index = file_name[index_S: -index_E] 
    return index

# Get file names
def Get_Location_Code(file_name):
    index_S = len(hd.Raw_Data_File_Path) + len(hd.Raw_Data_File_Name_Prefix) + len(hd.Raw_Data_File_Name_Start_Index) + 1
    len_location_code = len(hd.Colect_Location_Code_StartIndex)
    location_code = file_name[index_S: index_S + len_location_code] 
    return location_code

# Get file names
def Get_File_name_List(path):
    file_name_list= []
    for file_name in os.listdir(path):
        if(file_name.endswith(hd.Raw_Data_File_Name_Extension)):
            file_name_list.append(path + file_name)
    return file_name_list

#Import text from file
def Import_From_Text(file_name):
    with open(file_name, "r") as f:
        list = []
        for item in f:
            list.append(float(item.replace("\n","")))
    return list

#Import text from file and convert to int
def Import_From_Text_To_Int(file_name):
    with open(file_name, "r") as f:
        list = []
        for item in f:
            list.append(int(item.replace("\n","")))
    return list