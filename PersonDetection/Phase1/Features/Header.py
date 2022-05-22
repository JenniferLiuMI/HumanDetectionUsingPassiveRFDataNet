"""
Created by Bing Liu
Constants used in features related files
"""

import tensorflow as tf
import numpy as np

#Root folder
Root = 'e:/RFRoot/'

#Raw data file path
Raw_Data_File_Name_Prefix = 'RFRaw_'
Raw_Data_File_Name_Start_Index = '0000000'
Raw_Data_File_Name_Extension = '.pkl'
Raw_Data_File_Root= Root
Raw_Data_File_Path= Root + 'RFRawData/FullBand2019/'
Raw_Data_Log_File_Path = Root + 'RFDataLog/'

#File extension
Log_File_Extension = '.log'
JPG_File_Extension = ".png"
Raw_Data_Type = tf.float32
Raw_Data_Type_Str = 'float32'

#Model name
Data_Set_Train = "Train"
Data_Set_Valid = "Valid"
Data_Set_Test = "Test"

#Histogram
Hist_Bins = np.arange(0, 255, 5)
Hist_Bins_Len = len(Hist_Bins)

# Histogram Active Band
Hist_Model_Path_Active_Band = Root + 'Train/Hist/Band/Active/Models/'
Hist_Data_Set_Path_Active_Band = Root + 'Train/Hist/Band/Active/Data_Set/'

# Histogram Active Band Excluded Cell
Hist_Model_Path_Active_Band_ExcludedCell = Root + 'Train/Hist/Band/ActiveExcludedCell/Models/'
Hist_Data_Set_Path_Active_Band_ExcludedCell = Root + 'Train/Hist/Band/ActiveExcludedCell/Data_Set/'

# Histogram Inactive Band
Hist_Model_Path_Inactive_Band = Root + 'Train/Hist/Band/Inactive/Models/'
Hist_Data_Set_Path_Inactive_Band = Root + 'Train/Hist/Band/Inactive/Data_Set/'

# Histogram Inactive Band Excluded Cell
Hist_Model_Path_Inactive_Band_ExcludedCell = Root + 'Train/Hist/Band/InactiveExcludedCell/Models/'
Hist_Data_Set_Path_Inactive_Band_ExcludedCell = Root + 'Train/Hist/Band/InactiveExcludedCell/Data_Set/'

# Histogram Random Band Home
Hist_Model_Path_Random_Band_Home = Root + 'Train/Hist/Band/RandomHome/Models/'
Hist_Data_Set_Path_Random_Band_Home = Root + 'Train/Hist/Band/RandomHome/Data_Set/'

# Histogram Random Band Office
Hist_Model_Path_Random_Band_Office = Root + 'Train/Hist/Band/RandomOffice/Models/'
Hist_Data_Set_Path_Random_Band_Office = Root + 'Train/Hist/Band/RandomOffice/Data_Set/'

# Histogram Random Selective
Hist_Model_Path_Selective_Band = Root + 'Train/Hist/Band/Selective/Models/'
Hist_Data_Set_Path_Selective_Band = Root + 'Train/Hist/Band/Selective/Data_Set/'


# IQ data
IQ_Data_File_Name_Prefix = 'RFIQ_'

# IQ Active Band
IQ_Data_File_Path_Active_Band = Root + 'RFIQData/ActiveBand/'
IQ_Data_Set_Path_Active_Band = Root + 'Train/IQ/Band/Active/Data_Set/'
IQ_Model_Path_Active_Band = Root + 'Train/IQ/Band/Active/Models/'

# IQ Active Band ExcludedCell
IQ_Data_File_Path_Active_Band_ExcludedCell = Root + 'RFIQData/ActiveBandExcludedCell/'
IQ_Data_Set_Path_Active_Band_ExcludedCell = Root + 'Train/IQ/Band/ActiveExcludedCell/Data_Set/'
IQ_Model_Path_Active_Band_ExcludedCell = Root + 'Train/IQ/Band/ActiveExcludedCell/Models/'

# IQ Inactive Band
IQ_Data_File_Path_Inactive_Band = Root + 'RFIQData/InactiveBand/'
IQ_Data_Set_Path_Inactive_Band = Root + 'Train/IQ/Band/Inactive/Data_Set/'
IQ_Model_Path_Inactive_Band = Root + 'Train/IQ/Band/Inactive/Models/'

# IQ Inactive Band ExcludedCell
IQ_Data_File_Path_Inactive_Band_ExcludedCell = Root + 'RFIQData/InactiveBandExcludedCell/'
IQ_Data_Set_Path_Inactive_Band_ExcludedCell = Root + 'Train/IQ/Band/InactiveExcludedCell/Data_Set/'
IQ_Model_Path_Inactive_Band_ExcludedCell = Root + 'Train/IQ/Band/InactiveExcludedCell/Models/'

# IQ Random Band Home
IQ_Data_File_Path_Random_Band= Root + 'RFIQData/RandomBand/'
IQ_Data_Set_Path_Random_Band_Home = Root + 'Train/IQ/Band/RandomHome/Data_Set/'
IQ_Model_Path_Random_Band_Home = Root + 'Train/IQ/Band/RandomHome/Models/'

# IQ Random Band Office
IQ_Data_Set_Path_Random_Band_Office = Root + 'Train/IQ/Band/RandomOffice/Data_Set/'
IQ_Model_Path_Random_Band_Office = Root + 'Train/IQ/Band/RandomOffice/Models/'

# IQ Selective Band
IQ_Data_File_Path_Band_Selective= Root + '/RFIQData/SelectiveBand/'
IQ_Data_Set_Path_Band_Selective = Root + 'Train/IQ/Band/Selective/Data_Set/'
IQ_Model_Path_Band_Selective = Root + 'Train/IQ/Band/Selective/Models/'

# IQ Selective Band Home
IQ_Data_Set_Path_Band_Selective_Home = Root + 'Train/IQ/Band/SelectiveHome/Data_Set/'
IQ_Model_Path_Band_Selective_Home = Root + 'Train/IQ/Band/SelectiveHome/Models/'

# IQ Selective Band Office
IQ_Data_Set_Path_Band_Selective_Office = Root + 'Train/IQ/Band/SelectiveOffice/Data_Set/'
IQ_Model_Path_Band_Selective_Office = Root + 'Train/IQ/Band/SelectiveOffice/Models/'

# Amplitude Active Band
AMP_Data_Set_Path_Active_Band = Root + 'Train/Amplitude/Band/Active/Data_Set/'
AMP_Model_Path_Active_Band = Root + 'Train/Amplitude/Band/Active/Models/'

# Amplitude Active Band Excluded Cell
AMP_Data_Set_Path_Active_Band_ExcludedCell = Root + 'Train/Amplitude/Band/ActiveExcludedCell/Data_Set/'
AMP_Model_Path_Active_Band_ExcludedCell = Root + 'Train/Amplitude/Band/ActiveExcludedCell/Models/'

# Amplitude Inactive Band
AMP_Data_Set_Path_Inactive_Band = Root + 'Train/Amplitude/Band/Inactive/Data_Set/'
AMP_Model_Path_Inactive_Band = Root + 'Train/Amplitude/Band/Inactive/Models/'

# Amplitude Inactive Band Excluded Cell
AMP_Data_Set_Path_Inactive_Band_ExcludedCell = Root + 'Train/Amplitude/Band/InactiveExcludedCell/Data_Set/'
AMP_Model_Path_Inactive_Band_ExcludedCell = Root + 'Train/Amplitude/Band/InactiveExcludedCell/Models/'

# Amplitude Random Band Home
AMP_Data_Set_Path_Random_Band_Home = Root + 'Train/Amplitude/Band/RandomHome/Data_Set/'
AMP_Model_Path_Random_Band_Home = Root + 'Train/Amplitude/Band/RandomHome/Models/'

# Amplitude Random Band Office
AMP_Data_Set_Path_Random_Band_Office = Root + 'Train/Amplitude/Band/RandomOffice/Data_Set/'
AMP_Model_Path_Random_Band_Office = Root + 'Train/Amplitude/Band/RandomOffice/Models/'

# Amplitude Selective Band Home and Office
AMP_Data_Set_Path_Selective_Band = Root + 'Train/Amplitude/Band/Selective/Data_Set/'
AMP_Model_Path_Selective_Band = Root + 'Train/Amplitude/Band/Selective/Models/'

# Amplitude Selective Band Home
AMP_Data_Set_Path_Selective_Band_Home = Root + 'Train/Amplitude/Band/SelectiveHome/Data_Set/'
AMP_Model_Path_Selective_Band_Home = Root + 'Train/Amplitude/Band/SelectiveHome/Models/'

# Amplitude Selective Band Office
AMP_Data_Set_Path_Selective_Band_Office = Root + 'Train/Amplitude/Band/SelectiveOffice/Data_Set/'
AMP_Model_Path_Selective_Band_Office = Root + 'Train/Amplitude/Band/SelectiveOffice/Models/'

# IQ Random Band Home
IQ_Data_File_Path_Random_Band= Root + 'RFIQData/RandomBand/'
IQ_Data_Set_Path_Random_Band_Home = Root + 'Train/IQ/Band/RandomHome/Data_Set/'
IQ_Model_Path_Random_Band_Home = Root + 'Train/IQ/Band/RandomHome/Models/'

# Phase Active Band
Phase_Data_Set_Path_Active_Band = Root + 'Train/Phase/Band/Active/Data_Set/'
Phase_Model_Path_Active_Band = Root + 'Train/Phase/Band/Active/Models/'

# Phase Inactive Band
Phase_Data_Set_Path_Inactive_Band = Root + 'Train/Phase/Band/Inactive/Data_Set/'
Phase_Model_Path_Inactive_Band = Root + 'Train/Phase/Band/Inactive/Models/'

# FFT amplitude Active Band
FFT_Data_File_Path_Active_Band = Root + 'RFFFT/ActiveBand/'
FFT_AMP_Data_Set_Path_Active_Band = Root + 'Train/FFT_AMP/Band/Active/Data_Set/'
FFT_AMP_Model_Path_Active_Band = Root + 'Train/FFT_AMP/Band/Active/Models/'

#Data collection locations
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

