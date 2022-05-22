"""
Created by Bing Liu
Constants used in data processing
"""

#Root path
Root = 'e:/RFRoot/'

#Raw data file
Raw_Data_File_Name_Prefix = 'RFRaw_'
Raw_Data_File_Name_Start_Index = '0000000'
Raw_Data_File_Name_Extension = '.pkl'
Raw_Data_File_Root= Root
Raw_Data_File_Path= Root + 'RFRawData/'
Raw_Data_Active_Band= Root + 'RFRawData/ActiveBand/'
Raw_Data_PCASelectedBand= Root + 'RFRawData/PCASelectedBand/'
Raw_Data_Active_Band_ExcludedCell= Root + 'RFRawData/ActiveBandExcludedCell/'
Raw_Data_Inactive_Band= Root + 'RFRawData/InactiveBand/'
Raw_Data_Inactive_Band_ExcludedCell= Root + 'RFRawData/InactiveBandExcludedCell/'
Raw_Data_Random_Band= Root + 'RFRawData/RandomBand/'
Raw_Data_Full_Band= Root + 'RFRawData/FullBand/'
Raw_Data_Selective_Band= Root + 'RFRawData/SelectiveBand/'

#Frequency bands list file
Active_Band_Freq_List_File = 'RFRawData/FreqsList/ActiveBandFreqs.txt'
Active_Band_Freq_List_File_ExcludedCell = 'RFRawData/FreqsList/ActiveBandFreqsExcludedCell.txt'
Inactive_Band_Freq_List_File = 'RFRawData/FreqsList/InactiveBandFreqs.txt'
Inactive_Band_Freq_List_File_ExcludedCell = 'RFRawData/FreqsList/InactiveBandFreqsExcludedCell.txt'
Random_Band_Freq_List_File = 'RFRawData/FreqsList/RandomBandFreqs.txt'

#File extension
Log_File_Extension = '.log'
Text_File_Extension = '.txt'
CSV_File_Extension = '.csv'
JPG_File_Extension = '.jpg'
PNG_File_Extension = '.png'
Model_File_Extension = '.mod'

#IQ Data
IQ_Data_Type_Str = 'float32'
IQ_Data_File_Path= Root + 'RFIQData/'
IQ_Data_File_Name_Prefix = 'RFIQ_'
IQ_Data_Active_Band= Root + 'RFIQData/ActiveBand/'
IQ_Data_Active_Band_ExcludedCell= Root + 'RFIQData/ActiveBandExcludedCell/'
IQ_Data_Inactive_Band= Root + 'RFIQData/InactiveBand/'
IQ_Data_Inactive_Band_ExcludedCell= Root + 'RFIQData/InactiveBandExcludedCell/'
IQ_Data_Random_Band= Root + 'RFIQData/RandomBand/'
IQ_Data_Selective_Band= Root + 'RFIQData/SelectiveBand/'

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
Colect_Location_Code = '01'

#Occupancy status
Person_Present_No = 0
Person_Present_Yes = 1
Person_Present_Invalid = -1