"""
Created by Bing Liu
Human detection using active bands
"""

import Common as com
import Header as hd
import Raw2DCreateDataSet
import Raw2DCNNTrain
import Raw2DCNNPredict

# Configrations 
Model_Name = "CNN"
Create_Dataset = False
Training = False
Predict = True

# Dataset paths
Data_Set_Path = hd.Raw2D_Data_Set_Path_Active_Band
Raw_Data_File_Path = hd.Raw_Data_File_Path_Active_Band
Model_Path = hd.Raw2D_Model_Path_Active_Band

# Get the numbers of frequences
Freqs_Num = len(com.Load_Feq_List(hd.Root + hd.Freq_List_File_Active_Band ))

# Create dataset
if Create_Dataset == True:

    com.Remove_Data_File(Data_Set_Path, hd.Raw_Data_File_Name_Extension)

    Raw2DCreateDataSet.Create(Raw_Data_File_Path,
                              Data_Set_Path,
                              hd.Raw2D_Data_Set_Train,
                              hd.Raw2D_Data_Set_Valid,
                              hd.Raw2D_Data_Set_Test,
                              300,
                              ["01"],
                              Freqs_Num,
                              hd.Samples_Num)

# Train CNN
if Training == True:

    EPOCHS = 10
    BATCH_SIZE = 16
    Raw2DCNNTrain.Run(EPOCHS,
                      BATCH_SIZE,
                      Model_Path,
                      Data_Set_Path,
                      hd.Raw2D_Data_Set_Train,
                      hd.Raw2D_Data_Set_Valid,
                      hd.Raw2D_Data_Set_Test,
                      Freqs_Num,
                      hd.Samples_Num,
                      Model_Name)

# Prediction human occupancy
if Predict == True:

    Test_File = Data_Set_Path + hd.Raw2D_Data_Set_Test + hd.Raw_Data_File_Name_Extension
    Raw2DCNNPredict.Run(Test_File,
                        Model_Path,
                        Freqs_Num,
                        hd.Samples_Num,
                        Model_Name)

