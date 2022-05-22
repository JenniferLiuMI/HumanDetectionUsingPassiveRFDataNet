"""
Created by Bing Liu
Collect RF data using SDR
"""
import datetime
import logging
import SDRCommon as com
import SDRHeader as hd
import SDRDemodulateFMToAudio
from rtlsdr import RtlSdr
from rtlsdr import librtlsdr 
import SDRCaptureRawData as SDR
import time
import ctypes
import numpy as np
import time
import os
from tkinter import *
import threading 

#global variable
Person_Present_Status = hd.Person_Present_Yes
Person_Present_Status_Last = hd.Person_Present_Yes
Stop_Collecting = False
Thread_Is_Running = False
sdr = None

#Log information
com.Log_Setup()
 
#RF data collection
def Collect():
    global Person_Present_Status
    global Stop_Collecting
    global sdr
    global Thread_Is_Running
    global Person_Present_Status_Last
    Thread_Is_Running = True
    collect_times = 1

    time.sleep(10)
    while(collect_times<1000):

        if (Stop_Collecting == True):
            Thread_Is_Running = False
            #Delete last raw data file
            file = com.Get_Last_Raw_Data_File_name()
            os.remove(file)
            msg_arr.append('----------------------------------------------------')
            msg_arr.append(str(datetime.datetime.now()) + ' Collecting is stopped by user.')
            msg_arr.append("Deleted raw data file: " + file)
            com.Log_Info(msg_arr)
            msg_arr.clear()
            print("Deleted raw data file: " + file)
            time.sleep(10)
            break

        print('Start ' + str(collect_times))
        collect_times = collect_times+1
        raw_data_arry = []

        status_arr = []
        raw_data_arry = []
        status_change_arr = []
        msg_arr = []
        index = 0
        feq_arr = com.Generate_Feq_List();

        print("Person_Present_Status = :"  + str(Person_Present_Status))
        print("Person_Present_Status_Last = :"  + str(Person_Present_Status_Last))

        if (Person_Present_Status != Person_Present_Status_Last):
            #Status changed
            msg_arr.append('----------------------------------------------------')
            msg_arr.append("Status changed from {0:d} to {1:d}".format(Person_Present_Status_Last, Person_Present_Status))
            Person_Present_Status_Last = Person_Present_Status
            #Delete last raw data file and wait for 10 senonds
            file = com.Get_Last_Raw_Data_File_name()
            os.remove(file)
            msg_arr.append("Deleted raw data file: " + file)
            com.Log_Info(msg_arr)
            msg_arr.clear()
            print("Deleted raw data file: " + file)
            time.sleep(10);

        raw_data_file_name = com.Generate_Raw_Data_File_name(hd.Colect_Location_Code, Person_Present_Status)
        status_change_arr.append((str(Person_Present_Status_Last)  + ' ' + str(datetime.datetime.now())))
        #time_sleep = float(1.0 - float(hd.Cap_Milliseconds)/1000.0)
        N_Bytes = com.Cal_Samples_Num(hd.Sample_Rate, hd.Cap_Milliseconds)
        msg_arr.append('----------------------------------------------------')
        msg_arr.append(str(datetime.datetime.now()) + ' Start to collect raw data.')
        msg_arr.append("Colect location: " + hd.Colect_Locations.get(hd.Colect_Location_Code))
        msg_arr.append("Sample rate: {0:f}Hz".format(hd.Sample_Rate))
        msg_arr.append("Sample time: {0:d} milliseconds per frequency".format(hd.Cap_Milliseconds))
        msg_arr.append("Samples: {0:f} per frequency".format(N_Bytes ))
        msg_arr.append("Raw data file: " + raw_data_file_name)
        #msg_arr.append("Sleep time: {0:f}Seconds".format(time_sleep))
        com.Log_Info(msg_arr)
        msg_arr.clear()
        while(index < len(feq_arr)):
            status_arr.append(Person_Present_Status_Last)
            #Samples to capture
            SDR_bytes = SDR.Capture(sdr, feq_arr[index], hd.Sample_Rate, N_Bytes)
            raw_data_arry.append(np.copy(SDR_bytes))
            feq = str(feq_arr[index])
            print(feq)
            msg_arr.append("Feq: {0:f}Hz".format(feq_arr[index]))
            #time.sleep(time_sleep)
            index += 1

        for raw_data in raw_data_arry:
            #Dump raw data to file
            com.Dump_Raw_Data_To_File(raw_data_file_name,feq_arr,status_arr,raw_data_arry)

        #Log status chage
        com.Log_Status_Change(status_change_arr)
        msg_arr.append(str(datetime.datetime.now()) + ' Finished collecting raw data')
        com.Log_Info(msg_arr)
        msg_arr.clear()
    
    sdr.close()

#Set occupancy status
def Person_Present():
    global Person_Present_Status
    Person_Present_Status = hd.Person_Present_Yes
    if(Thread_Is_Running== False):
        Button_Start.config(state="active")

#Set unoccupancy status
def Person_Absent():
    global Person_Present_Status
    Person_Present_Status = hd.Person_Present_No
    if(Thread_Is_Running== False):
        Button_Start.config(state="active")

#Start collecting
def Start():
    #Open SDR
    global sdr
    global Thread_Is_Running
    while(Thread_Is_Running == True):
        time.sleep(1)
    sdr = RtlSdr(1)
    t1 = threading.Thread(target=Collect) 
    t1.start()
    Button_Start.config(state="disabled")
    Button_Stop.config(state="active")
    
#Stop collecting
def Stop():
    global Stop_Collecting
    global sdr
    global Thread_Is_Running
    Stop_Collecting = True
    while(Thread_Is_Running):
        time.sleep(1)
    #Close SDR
    sdr.close()
    Stop_Collecting = False
    Thread_Is_Running = False
    Button_Start.config(state="active")
    Button_Stop.config(state="disabled")

#Defind userface
root = Tk() 
frame = Frame(root) 
frame.pack() 
bottomframe = Frame(root) 
bottomframe.pack( side = BOTTOM ) 
Button_Start = Button(frame, text = 'Start', fg='green', width=25, command=Start, state=DISABLED) 
Button_Start.pack( side = LEFT ) 
Button_Stop = Button(frame, text = 'Stop', fg ='red', width=25, command=Stop, state=DISABLED) 
Button_Stop.pack( side = LEFT) 
v = IntVar()
Radio_Absent = Radiobutton(frame, text="Person Absent: 0", fg='red', variable=v, value=2, width=25,  command=Person_Absent).pack(anchor=W)
Radio_Present = Radiobutton(frame, text="Person Present: 1", fg='green', variable=v, value=1, width=25,  command=Person_Present).pack(anchor=W)
root.mainloop() 