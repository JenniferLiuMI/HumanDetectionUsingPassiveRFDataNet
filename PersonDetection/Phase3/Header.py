"""
Created by Bing Liu
Constants used in phase 3
"""

import tensorflow as tf
import numpy as np
import platform

#Root path
Root = 'e:/RFRoot/'

#Raw data configs
Raw_Data_File_Name_Start_Index = '0000000'
Raw_Data_File_Name_Prefix = 'RFRaw_'
Raw_Data_File_Name_Extension = '.pkl'
Raw_Data_File_Root= Root
Raw_Data_File_Path= Root + 'RFRawData/FullBand2019/'
Raw_Data_Log_File_Path = Root + 'RFDataLog/'
Raw_Data_Type = tf.float32
Raw_Data_Type_Str = 'float32'

#IQ data configs
IQ_Data_Type = tf.float32
IQ_Data_Type_NP = np.float64
IQ_Data_Type_Str = 'float32'

#Other configs
Power_Avg_Data_File_Name_Prefix = 'RFPower_Avg_'
JPG_File_Extension = ".png"

#GAN paths
GAN_Root = Root + 'Phase3/GAN/'
GAN_Raw2D_Data_Set_Path_Band = GAN_Root + 'Raw/Bands_{0}_Samples_{1}/'
GAN_Raw2D_Data_Set_Path_Active_Band = GAN_Root + 'Raw/Active/Raw2D_Data_Set/'
GAN_Raw2D_Model_Path_Active_Band = GAN_Root + 'Raw/Active/Raw2D_Models/'
GAN_Power_Avg_Path = GAN_Root + 'Power_Avg/All/'
GAN_Power_Avg_File_List_Location_01_Hour_From_6_To_22 = GAN_Power_Avg_Path + "File_List_Location_01_Hour_From_6_To_22.txt"
GAN_Power_Avg_File_List_Location_05_Hour_From_6_To_22 = GAN_Power_Avg_Path + "File_List_Location_05_Hour_From_6_To_22.txt"
GAN_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22 = GAN_Power_Avg_Path + "File_List_Location_01_05_Hour_From_6_To_22.txt"
GAN_Power_Avg_File_List_Location_01_Hour_From_6_To_22_Raw = GAN_Power_Avg_Path + "File_List_Location_01_Hour_From_6_To_22_Raw.txt"
GAN_Power_Avg_File_List_Location_01_05_Hour_From_6_To_22_Raw = GAN_Power_Avg_Path + "File_List_Location_01_05_Hour_From_6_To_22_Raw.txt"
GAN_Power_Avg_Data_Set_Path_01 = GAN_Root + 'Power_Avg/Data_Set_01/'
GAN_Power_Avg_Data_Set_Path_05 = GAN_Root + 'Power_Avg/Data_Set_05/'
GAN_Power_Avg_Data_Set_Path_01_05 = GAN_Root + 'Power_Avg/Data_Set_01_05/'
Full_Band_Freq_List_File = GAN_Power_Avg_Path + 'FullBandFreqList.txt'
Band_Selection_File_Path = GAN_Root + 'BandSelection/'

#GAN PAC selected bands
PCA_Rank_List_File =  Band_Selection_File_Path + 'PCARankList_{}.txt'
PCA_Selected_Band_List_File = Band_Selection_File_Path + 'PCASelectedFreqList_{}.txt'
PCA_Selected_Band_List_File_Sorted = Band_Selection_File_Path + 'PCASelectedFreqListSorted_{}.txt'

#GAN RFE selected bands
Freq_List_File_RFE_Seleted_Band = 'RFRawData/FreqsList/RFESeletedBandFreqs.txt'
RFE_Rank_List_File =  Band_Selection_File_Path + 'RFERankList_{}.txt'
RFE_Selected_Band_List_File = Band_Selection_File_Path + 'RFESelectedFreqList_{}.txt'

#Dataset file name
Data_Set_Train = "Train"
Data_Set_Valid = "Valid"
Data_Set_Test = "Test"

# IQ data
IQ_Data_File_Name_Prefix = 'RFIQ_'

#Data collection locations
Colect_Locations =	{
  "01": "Home1 Antenna postion1",
  "02": "OU EC 423 Meeting Room",
  "03": "120A Dodge Hall",
  "04": "OU EC 279 Class Room",
  "05": "OU EC 422 Office",
  "06": "Car Rear Seat Left",
  "07": "Home2 Huaizheng",
  "08": "Car Driver Seat",
  "09": "Car Rear Seat Right",
  "10": "Car2 Bing",
  "11": "Home1 postion3 far not use",
  "12": "Home1 postion2 far",
  "13": "Home2 postion2 far",
}

#Occupancy status
Person_Present_No = 0
Person_Present_Yes = 1
Person_Present_Invalid = -1