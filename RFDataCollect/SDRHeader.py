"""
Created by Bing Liu
Constants used in data collection
"""

#SDR configs
Sample_Rate = 2.4e6
Cap_Milliseconds = 2

#Root path
Root = 'e:/RFRoot/'

#Raw data
Raw_Data_File_Name_Prefix = 'RFRaw_'
Raw_Data_File_Name_Start_Index = '0000000'
Raw_Data_File_Name_Extension = '.pkl'
Raw_Data_File_Root= Root
Raw_Data_File_Path= Root + 'RFRawData/'
Raw_Data_Log_File_Path = Root + 'RFDataLog/'
Raw_Data_File_Max_Size = 500e6

#Collection configs
Colect_Location_Code_StartIndex = '00'
Log_File_Extension = '.log'
Colect_Location_Code = '01'
Pack_File_Filters =	["01"]

#Master files
Master_File_Header_File_Name = 'DataFileName'
Master_File_Header_Location = 'Location'
Master_File_Header_Time_Stemp = 'Time_Stemp'

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