"""
Created by Bing Liu
The common information used in phase 2
"""

#Root folders
Root = 'e:/RFRoot/'
Phase2_Root_Path = Root + 'Phase2/'
Phase2_Power_Path = Phase2_Root_Path + "Power/"

#File paths
Raw_Data_File_Name_Prefix = 'RFRaw_'
Raw_Data_File_Name_Start_Index = '0000000'
Colect_Location_Code_StartIndex = '00'
Raw_Data_File_Name_Extension = '.pkl'
Raw_Data_File_Root= Root
Raw_Data_File_Path= Root + 'RFRawData/'
Raw_Data_PCASelectedBand= Root + 'RFRawData/PCASelectedBand/'
Raw_Data_Full_Band= Root + 'RFRawData/FullBand/'

#PCA
PCA_Path = Root + 'Phase2/PCA/'
PCA_Path_Classification = PCA_Path + 'Classification'

#RFE
RFE_Path = Root + 'Phase2/RFE/'
RFE_Path_Classification = RFE_Path + 'Classification'

#File extension name
Log_File_Extension = '.log'
Text_File_Extension = '.txt'
JPG_File_Extension = '.jpg'
PNG_File_Extension = '.png'
Model_File_Extension = '.mod'

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