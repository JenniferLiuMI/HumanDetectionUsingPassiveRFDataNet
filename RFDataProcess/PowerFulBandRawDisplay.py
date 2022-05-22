"""
Created by Bing Liu
Plot powers using raw data
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
import matplotlib.pyplot as plt
import AMPD
import PowerChartDisplay
window=10

#Caculate the powers of frequency bands
def Get_Freqs_Powers(raw_file_list):

    freqs = []
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
    return freqs, power_arry

#Select the frequency bands
def Pick_Freq(step, percent, power_arry_len, power_arry):
    peak_list=[]
    end = 0
    for i in range(0, power_arry_len, step):
        if (i+step)%step == 0:
            end = i+step
        else:
            end = power_arry_len - 1
        power_mean = np.mean(power_arry[i:end], axis=0)
        peaks = (np.zeros((len(freqs)))).astype(int)
        peaked = AMPD.find_peaks_adaptive(power_mean, window)
        peaks[peaked] = 1
        peak_list.append(peaks)

    peak_arry = np.asarray(peak_list)
    peak_sum = np.sum(peak_arry, axis=0)
    
    threadhold =  int(len(peak_list) * percent)
    count = 0
    picked = (np.zeros((len(freqs)))).astype(int)

    peak_sum[peak_sum<threadhold] = 0
    peak_sum[peak_sum>=threadhold] = 1

    picked = (peak_sum).tolist()
    pks = freqs[np.flatnonzero(picked)]
    print("Total " + str(len(pks)) + "picked. Outputted freqs into file freq.txt")
    np.savetxt('freq.txt',pks, delimiter=',')

    return picked

#Entrance of execution
if __name__ == '__main__':

    file_list = []
    power_list = []
    power_mean_list = []
    picked_list = []
    name_list = []
    title = ""

    file_list = com.Get_File_name_List(hd.Raw_Data_File_Path_Ful_No_Status)
    freqs, power_arry = Get_Freqs_Powers(file_list)
    power_arry_len = len(power_arry)

    power_mean_all = np.mean(power_arry, axis=0)

    percent = 0.5
    step = 5
    picked = Pick_Freq(step, percent, power_arry_len, power_arry)
    picked_list.append(picked)
    name = "Threshold: " + str(percent) + " Step: " + str(step) + " Freq: " + str(np.sum(picked))
    name_list.append(name)
    title += name + '\n'
    
    color_list = []
    color_list.append('blue')

    PowerChart.Display(freqs, power_mean_all, picked_list, title, name_list, color_list)