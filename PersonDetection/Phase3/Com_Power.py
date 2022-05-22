"""
Created by Bing Liu
Common functions to process power data
"""

import pickle
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import Header as hd
import Common as com

Lable_fontsize=10
legend_fontsize=10
ticks_fontsize=10
Freq_X_Limit_Low = 24
Freq_X_Limit_High = 1760

def Gene_Power_File_List( Location_Filters):
    files_list = []
    data_files = com.Get_File_name_List(hd.Raw_Data_File_Path)
    for file in data_files:
       location_code = com.Get_Location_Code(file, hd.Raw_Data_File_Path)
       for code in Location_Filters:
           if code == location_code: 
                files_list.append(file)
                break
    raw_data_file_list = com.Get_File_List_By_Hour(data_files, 6, 22)
    with open(hd.GAN_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22_Raw, 'wb') as f:
        pickle.dump(raw_data_file_list, f, pickle.HIGHEST_PROTOCOL)

    with open(hd.GAN_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22_Raw, "rb")as f:
        raw_data_file_list = pickle.load(f)

    power_age_data_file_list = []
    for raw_data_file_name in raw_data_file_list:
        power_avg_file_name = raw_data_file_name.replace(hd.Raw_Data_File_Path, hd.GAN_Power_Avg_Path )
        power_avg_file_name = power_avg_file_name.replace(hd.Raw_Data_File_Name_Prefix, hd.Power_Avg_Data_File_Name_Prefix )
        power_age_data_file_list.append(power_avg_file_name)

    with open(hd.GAN_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22, 'wb') as f:
        pickle.dump(power_age_data_file_list, f, pickle.HIGHEST_PROTOCOL)

    with open(hd.GAN_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22, "rb")as f:
        power_avg_file_name = pickle.load(f)

    power_avg_file_name_05=[]
    for file in power_avg_file_name:
        if file.find('_05_')>0:
            power_avg_file_name_05.append(file)

    with open(hd.GAN_Power_Avg_File_List_Location_05_Hour_From_6_To_22, 'wb') as f:
        pickle.dump(power_avg_file_name_05, f, pickle.HIGHEST_PROTOCOL)
#Gene_Power_File_List("05")

def Caculate_Freqs_Powers(Raw_Data):

    raw_data_per_file = Raw_Data/127.5 - 1
    power_per_file = np.linalg.norm(raw_data_per_file, axis=1)
    power_per_file = power_per_file*power_per_file
    db = 10*np.log10(power_per_file/len(raw_data_per_file[0])/2)
    freq_power = np.asarray(db)
    return freq_power.reshape(len(freq_power))

def Caculate_Freqs_Powers_Arr(Raw_Data_Arr):

    freq_powers = np.zeros((np.shape(Raw_Data_Arr)[0],np.shape(Raw_Data_Arr)[1]), dtype = hd.Raw_Data_Type_Str)
    i = 0
    for raw_data in Raw_Data_Arr:
        freq_power =  Caculate_Freqs_Powers(raw_data)
        freq_powers[i] = freq_power
        i = i + 1
    return freq_powers

def Caculate_Freqs_Powers_Avg(Raw_Data_Arr):

    power_list_all = []
    for i in range(len(Raw_Data_Arr)):
        raw_data_per_file = Raw_Data_Arr[i]/127.5 - 1
        power_per_file = np.linalg.norm(raw_data_per_file, axis=1)
        power_per_file = power_per_file*power_per_file
        db = 10*np.log10(power_per_file/len(raw_data_per_file[0])/2)
        power_list_all.append(db)
    
    power_arry = np.asarray(power_list_all)
    power_mean = np.mean(power_arry, axis=0)

    return power_mean

def Get_Freqs_Power_Ave(Raw_Train, Label_Train, Raw_Valid, Label_Valid):

    raw_data = np.concatenate((Raw_Train, Raw_Valid), axis=0)
    labels = np.concatenate((Label_Train, Label_Valid), axis=0)

    label_indices_Yes = np.where(labels == hd.Person_Present_Yes)
    label_indices_No = np.where(labels == hd.Person_Present_No)

    raw_data_Yes = raw_data[np.array(label_indices_Yes[0])]
    raw_data_No = raw_data[np.array(label_indices_No[0])]

    power_avg_Yes = Caculate_Freqs_Powers_Avg(raw_data_Yes)
    power_avg_No = Caculate_Freqs_Powers_Avg(raw_data_No)

    return np.reshape(power_avg_Yes, len(power_avg_Yes)), np.reshape(power_avg_No, len(power_avg_No))

def Get_Freqs_Power_Ave_By_Power(Power_Train, Label_Train, Power_Valid, Label_Valid):

    power = np.concatenate((Power_Train, Power_Valid), axis=0)
    labels = np.concatenate((Label_Train, Label_Valid), axis=0)

    label_indices_Yes = np.where(labels == hd.Person_Present_Yes)
    label_indices_No = np.where(labels == hd.Person_Present_No)

    power_Yes = power[np.array(label_indices_Yes[0])]
    power_No = power[np.array(label_indices_No[0])]

    power_avg_Yes = np.average(power_Yes, axis=0)
    power_avg_No = np.average(power_No, axis=0)

    return power_avg_Yes,power_avg_No

def Get_Freqs_Power_Ave_By_Labels(Power_Train, Label_Train, Power_Valid, Label_Valid):

    power = np.concatenate((Power_Train, Power_Valid), axis=0)
    labels = np.concatenate((Label_Train, Label_Valid), axis=0)

    label_indices_Yes = np.where(labels == hd.Person_Present_Yes)
    label_indices_No = np.where(labels == hd.Person_Present_No)

    power_Yes = power[np.array(label_indices_Yes[0])]
    power_No = power[np.array(label_indices_No[0])]

    return power_Yes,power_No


def Export_Power_Avg_All():
    raw_data_file_list = com.Get_File_List(hd.Raw_Data_File_Path, hd.Raw_Data_File_Name_Extension)
    for raw_data_file_name in raw_data_file_list:
        raw_file = open(raw_data_file_name, "rb")
        freq_arr = pickle.load(raw_file)
        lables_per_file = pickle.load(raw_file)
        raw_data_per_file = pickle.load(raw_file)
        freq_powers_per_file = Com_Power.Caculate_Freqs_Powers(raw_data_per_file)
        power_avg_file_name = raw_data_file_name.replace(hd.Raw_Data_File_Path, hd.GAN_Power_Avg_Path )
        power_avg_file_name = power_avg_file_name.replace(hd.Raw_Data_File_Name_Prefix, hd.Power_Avg_Data_File_Name_Prefix )
        with open(power_avg_file_name, 'wb') as f:
            print("dump to : {}".format(power_avg_file_name))
            pickle.dump((freq_arr, lables_per_file, freq_powers_per_file), f, pickle.HIGHEST_PROTOCOL)

def Get_Power_Avg_File_List_Location_01_Hour_From_6_To_22():
    file = open(hd.GAN_Power_Avg_File_List_Location_01_Hour_From_6_To_22, "rb")
    return pickle.load(file)

def Get_Power_Avg_File_List_Location_05_Hour_From_6_To_22():
    file = open(hd.GAN_Power_Avg_File_List_Location_05_Hour_From_6_To_22, "rb")
    return pickle.load(file)

def Get_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22():
    file = open(hd.GAN_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22, "rb")
    return pickle.load(file)

def Get_Power_Avg_List_By_Location(Location_code):
    power_avg_file_list = com.Get_File_List_By_Location(hd.GAN_Power_Avg_Path, hd.Raw_Data_File_Name_Extension, Location_code)
    file_list = []
    for power_avg_file in power_avg_file_list:
        file = power_avg_file.replace(hd.GAN_Power_Avg_Path, hd.Raw_Data_File_Path)
        file = file.replace(hd.Power_Avg_Data_File_Name_Prefix, hd.Raw_Data_File_Name_Prefix)
        file_list.append(file)
    return file_list

def Get_Power_Avg_List(File_Num):
    power_avg_file_list = Get_Power_Avg_File_List_Location_01_Hour_From_6_To_22()
    power_avg_file_list = shuffle(power_avg_file_list)

    file_list_yes = []
    index = 0
    for power_avg_file_name in power_avg_file_list:
        if com.Get_Person_Present_Status(power_avg_file_name) == hd.Person_Present_Yes:
            file_list_yes.append(power_avg_file_name)
            index = index + 1    
            if index == File_Num:
                break

    file_list_no = []
    index = 0
    for power_avg_file_name in power_avg_file_list:
        if com.Get_Person_Present_Status(power_avg_file_name) == hd.Person_Present_No:
            file_list_no.append(power_avg_file_name)
            index = index + 1    
            if index == File_Num:
                break

    return file_list_yes, file_list_no

def Draw_Freqs_Powers_By_Raw_Sub_Plot(Figure_File_Name, Freqs, Raw_Data_arr):
    freq_powers_arr = Caculate_Freqs_Powers_Arr(Raw_Data_arr*127)
    Draw_Freqs_Figure_Single(Figure_File_Name, Freqs, freq_powers_arr)

def Draw_Freqs_Powers_Single_By_Power_Avg(Figure_File_Name, Freqs, Power_Avg_Arr):
    shape = np.shape(Power_Avg_Arr)
    Draw_Freqs_Figure_Single(Figure_File_Name, Freqs, np.reshape(Power_Avg_Arr, (shape[0], shape[1]*shape[2])))

def Draw_Freqs_Powers_Labels_By_Raw(Figure_File_Name, Freqs, Raw_Data_arr, Labels):
    freq_powers_arr = Caculate_Freqs_Powers_Arr(Raw_Data_arr*255)
    Draw_Freqs_Powers_Avg_Test(Figure_File_Name, Freqs, freq_powers_arr, Labels)

def Draw_Freqs_Powers_Labels_By_Power_Avg(Figure_File_Name, Freqs, Power_Avg_Arr, Labels):
    shape = np.shape(Power_Avg_Arr)
    Draw_Freqs_Powers_Avg_Test(Figure_File_Name, Freqs, np.reshape(Power_Avg_Arr, (shape[0], shape[1]*shape[2])), Labels)

def Draw_Freqs_Powers_Avg_Test(Figure_File_Name, Freqs, freq_powers_arr, Labels):
    label_indices_Yes = np.where(Labels == hd.Person_Present_Yes)
    label_indices_No = np.where(Labels == hd.Person_Present_No)

    freqs_powers_Yes = freq_powers_arr[np.array(label_indices_Yes[0])]
    freqs_powers_No = freq_powers_arr[np.array(label_indices_No[0])]

    Draw_Freqs_Figure_Both(Figure_File_Name, Freqs, np.mean(freqs_powers_Yes, axis=0), np.mean(freqs_powers_No, axis=0))

def Draw_Freqs_Powers_Avg_Train(Figure_File_Name, Freqs, x_train_all, y_train_all, x_valid_all, y_valid_all):

    freqs_powers_Yes, freqs_powers_No = Get_Freqs_Power_Ave(x_train_all, y_train_all, x_valid_all, y_valid_all)
    Draw_Freqs_Figure_Both(Figure_File_Name, Freqs, freqs_powers_Yes, freqs_powers_No)

def Draw_Freqs_Powers_Avg_Train_By_Power(Figure_File_Name, Freqs, x_train_all, y_train_all, x_valid_all, y_valid_all):

    freqs_powers_Yes, freqs_powers_No = Get_Freqs_Power_Ave_By_Power(x_train_all, y_train_all, x_valid_all, y_valid_all)
    Draw_Freqs_Figure_Both(Figure_File_Name, Freqs, freqs_powers_Yes, freqs_powers_No)

def Draw_Freqs_Figure_Single(Figure_File_Name, Freqs, Freqs_Powers_Arr):
    
    fig = plt.figure()
    for freq_power in Freqs_Powers_Arr:
        freq_power = freq_power.reshape(-1)
        plt.plot( Freqs, freq_power, linewidth=0.5)
    #plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    #plt.xticks(fontsize=ticks_fontsize)
    #plt.yticks(fontsize=ticks_fontsize)
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()   
    plt.savefig(Figure_File_Name, dpi =400)
    #plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
    #plt.ylim(-45, -5)
    #plt.show()
    plt.close()

def Draw_Freqs_Figure_Single_Sub_Plot(Figure_File_Name, Freqs, Freqs_Powers_Arr, Row_Num, Col_Num):
    
    fig = plt.figure()
    for i in range(1, len(Freqs_Powers_Arr)+1):
        plt.subplot(Row_Num, Col_Num, i)
        freq_power = Freqs_Powers_Arr[i-1].reshape(-1)
        plt.plot( Freqs, freq_power, linewidth=0.5)
        #plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
        #plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
        #plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()   
        
        #plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
        #plt.ylim(-45, -5)
        #plt.show()
    plt.savefig(Figure_File_Name, dpi =400)        
    plt.close()

def Draw_Freqs_Figure_Single_Sub_Plot_Label(Figure_File_Name, Freqs, Freqs_Powers_Arr, Label_Arr, Row_Num, Col_Num, line_color='blue'):
    
    fig = plt.figure()
    for i in range(1, len(Freqs_Powers_Arr)+1):
        plt.subplot(Row_Num, Col_Num, i)
        label = -1
        if len(np.shape(Label_Arr))==2:
            label = np.where(Label_Arr[i-1]>0)[0][0]
        else:
            label = Label_Arr[i-1]
        plt.suptitle( com.Get_Label_Name(label ), fontsize=2)
        freq_power = Freqs_Powers_Arr[i-1].reshape(-1)
        line_color = ''
        if label == hd.Person_Present_No:
            line_color = 'royalblue'
        else:
            line_color = 'red'
        plt.plot( Freqs, freq_power, linewidth=0.5, color=line_color)
        #plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
        plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
        plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()   
        
        #plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
        #plt.ylim(-45, -5)
        #plt.show()
    plt.savefig(Figure_File_Name, dpi =400)        
    plt.close()

def Draw_Freqs_Figure_One(Figure_File_Name, Freqs, Freqs_Powers_Arr, line_color='blue', my_figsize=(8, 6)):
    
    fig = plt.figure(figsize= my_figsize)
    ax = fig.add_subplot(111)
    ax.plot( Freqs, Freqs_Powers_Arr, linewidth=0.5, color=line_color)
    #plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    #plt.xticks(fontsize=ticks_fontsize)
    #plt.yticks(fontsize=ticks_fontsize)
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()   
    plt.savefig(Figure_File_Name, dpi =400)
    #plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
    #plt.ylim(-45, -5)
    #plt.show()
    plt.close()

def Draw_Freqs_Figure_Both(Figure_File_Name, Freqs, Freqs_Powers_Yes, Freqs_Powers_No):

    fig = plt.figure()
    plt.plot( Freqs, Freqs_Powers_Yes, color='red', linewidth=0.5, label='Occupied')
    plt.plot( Freqs, Freqs_Powers_No, color='blue', linewidth=0.5, label='Unoccupied')
    plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    #plt.xticks(fontsize=ticks_fontsize)
    #plt.yticks(fontsize=ticks_fontsize)
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()   
    Figure_File_Name = Figure_File_Name.replace("//","/")
    Figure_File_Name = Figure_File_Name.replace("./","/")
    plt.savefig(Figure_File_Name, dpi =400)
    #plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
    #plt.ylim(-45, -5)
    #plt.show()
    plt.close()