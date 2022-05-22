"""
Created by Bing Liu
The configuration information used in phase 1
"""
import tensorflow as tf
import numpy as np

#Root path
Root = 'e:/RFRoot/'

#Raw data file
Raw_Data_File_Name_Prefix = 'RFRaw_'
Raw_Data_File_Name_Start_Index = '0000000'
Raw_Data_File_Name_Extension = '.pkl'
Raw_Data_File_Root= Root
Raw_Data_File_Path= Root + 'RFRawData/RFRawData_197/'
Raw_Data_Log_File_Path = Root + 'RFDataLog/'

#Log file extension
Log_File_Extension = '.log'

#IQ data
IQ_Data_File_Path= Root + 'IQData/'
IQ_Data_File_Name_Prefix = 'RFIQ_'
IQ_Train = "IQ_Train"
IQ_Valid = "IQ_Valid"
IQ_Test = "IQ_Test"
IQ_Train_Path = Root + 'Training/'
IQ_Model_Path = Root + 'Saved_Models/'
IQ_Data_Type = tf.float32
IQ_Data_Type_Str = 'float32'

#Raw data
Raw_Train_Path = Root + 'Raw_Training/'
Raw_Model_Path = Root + 'Raw_Saved_Models/'
Raw_Train = "Raw_Train"
Raw_Valid = "Raw_Valid"
Raw_Test = "Raw_Test"
Raw_Data_Type = tf.float32
Raw_Data_Type_Str = 'float32'
Raw2D_Model_Path = Root + 'Location_01_05/Raw2D_Models/'
Raw2D_Data_Set_Path = Root + 'Location_01/Raw2D_Data_Set/'
Raw_Data_Full_Band= Root + 'RFRawData/FullBand/'
Raw2D_Test_New = "Raw2D_Test_New"
Raw_Data_New_File_Path= Root + 'RFRawDataNew/'
Raw2D_Data_Set_Train = "Raw2D_Train"
Raw2D_Data_Set_Valid = "Raw2D_Valid"
Raw2D_Data_Set_Test = "Raw2D_Test"

#Single Home
Raw2D_Data_Set_Path_Single_Home = Root + 'Train/Raw/Location/SingleHome/Raw2D_Data_Set/'
Raw2D_Model_Path_Single_Home = Root + 'Train/Raw/Location/SingleHome/Raw2D_Models/'

#Single Office
Raw2D_Data_Set_Path_Single_Office = Root + 'Train/Raw/Location/SingleOffice/Raw2D_Data_Set/'
Raw2D_Model_Path_Single_Office = Root + 'Train/Raw/Location/SingleOffice/Raw2D_Models/'

#Location Mix
Raw2D_Data_Set_Path_Mix = Root + 'Train/Raw/Location/Mix/Raw2D_Data_Set/'
Raw2D_Model_Path_Mix = Root + 'Train/Raw/Location/Mix/Raw2D_Models/'

#Active Band
Raw_Data_File_Path_Active_Band= Root + 'RFRawData/ActiveBand/'
Freq_List_File_Active_Band = 'RFRawData/FreqsList/ActiveBandFreqs.txt'
Raw2D_Model_Path_Active_Band = Root + 'Train/Raw/Band/Active/Raw2D_Models/'
Raw2D_Data_Set_Path_Active_Band = Root + 'Train/Raw/Band/Active/Raw2D_Data_Set/'

#Active Band ExcludedCell
Raw_Data_File_Path_Active_Band_ExcludedCell= Root + 'RFRawData/ActiveBandExcludedCell/'
Freq_List_File_Active_Band_ExcludedCell = 'RFRawData/FreqsList/ActiveBandFreqsExcludedCell.txt'
Raw2D_Model_Path_Active_Band_ExcludedCell = Root + 'Train/Raw/Band/ActiveExcludedCell/Raw2D_Models/'
Raw2D_Data_Set_Path_Active_Band_ExcludedCell = Root + 'Train/Raw/Band/ActiveExcludedCell/Raw2D_Data_Set/'

#Random Band Home
Raw_Data_File_Path_Random_Band= Root + 'RFRawData/RandomBand/'
Freq_List_File_Random_Band = 'RFRawData/FreqsList/RandomBandFreqs.txt'
Raw2D_Model_Path_Random_Band_Home = Root + 'Train/Raw/Band/RandomHome/Raw2D_Models/'
Raw2D_Data_Set_Path_Random_Band_Home = Root + 'Train/Raw/Band/RandomHome/Raw2D_Data_Set/'

#Random Band Office
Raw2D_Model_Path_Random_Band_Office = Root + 'Train/Raw/Band/RandomOffice/Raw2D_Models/'
Raw2D_Data_Set_Path_Random_Band_Office = Root + 'Train/Raw/Band/RandomOffice/Raw2D_Data_Set/'

#Inactive Band
Raw_Data_File_Path_Inactive_Band= Root + 'RFRawData/InactiveBand/'
Freq_List_File_Inactive_Band = 'RFRawData/FreqsList/InactiveBandFreqs.txt'
Raw2D_Model_Path_Inactive_Band = Root + 'Train/Raw/Band/Inactive/Raw2D_Models/'
Raw2D_Data_Set_Path_Inactive_Band = Root + 'Train/Raw/Band/Inactive/Raw2D_Data_Set/'

#Inactive Band Excluded Cell
Raw_Data_File_Path_Inactive_Band_ExcludedCell= Root + 'RFRawData/InactiveBandExcludedCell/'
Freq_List_File_Inactive_Band_ExcludedCell = 'RFRawData/FreqsList/InactiveBandFreqsExcludedCell.txt'
Raw2D_Model_Path_Inactive_Band_ExcludedCell = Root + 'Train/Raw/Band/InactiveExcludedCell/Raw2D_Models/'
Raw2D_Data_Set_Path_Inactive_Band_ExcludedCell = Root + 'Train/Raw/Band/InactiveExcludedCell/Raw2D_Data_Set/'

#Selective Band
Raw_Data_File_Path_Selective_Band= Root + '/RFRawData/SelectiveBand/'
Raw2D_Data_Set_Path_Selective_Band = Root + 'Train/Raw/Band/Selective/Raw2D_Data_Set/'
Raw2D_Model_Path_Selective_Band = Root + 'Train/Raw/Band/Selective/Raw2D_Models/'

#Time
Raw2D_Data_Set_Path_Time = Root + 'Train/Raw/Time/Raw2D_Data_Set/'
Raw2D_Model_Path_Time = Root + 'Train/Raw/Time/Raw2D_Models/'
Time_Period_Num = 4

#Dataset file name
Data_Set_Train = "Train"
Data_Set_Valid = "Valid"
Data_Set_Test = "Test"

#Data collection location
Colect_Locations =	{
  "01": "Bing Study Room",
  "02": "OU EC 423 Meeting Room",
  "03": "120A Dodge Hall",
  "04": "OU EC 279 Class Room",
  "05": "OU EC 422 Office",
  "06": "Bing Bed Room"
}

#Occupancy status
Person_Present_No = 0
Person_Present_Yes = 1
Person_Present_Invalid = -1