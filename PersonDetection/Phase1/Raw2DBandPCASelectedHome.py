"""
Created by Bing Liu
Human detection using selected bands
"""

import Common as com
import Header as hd
import Raw2DCreateDataSet
import Raw2DCNNTrain
import Raw2DCNNPredict

Model_Name = "CNN"
Create_Dataset = True
Training = True
Predict = False

Raw_Data_File_Path = hd.Raw_Data_File_Path_PCASelected_Band
Data_Set_Path = hd.Raw2D_Data_Set_Path_PCASelected_Band_Home
Model_Path = hd.Raw2D_Model_Path_PCASelected_Band_Home
Freqs_Num = len(com.Load_Feq_List(hd.Root + hd.Freq_List_File_PCASelected_Band ))
Sample_Num = hd.Samples_Num

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
                              Sample_Num)

if Training == True:

    EPOCHS = 20
    BATCH_SIZE = 16
    Raw2DCNNTrain.Run(EPOCHS,
                      BATCH_SIZE,
                      Model_Path,
                      Data_Set_Path,
                      hd.Raw2D_Data_Set_Train,
                      hd.Raw2D_Data_Set_Valid,
                      hd.Raw2D_Data_Set_Test,
                      Freqs_Num,
                      Sample_Num,
                      Model_Name)

if Predict == True:

    Test_File = Data_Set_Path + hd.Raw2D_Data_Set_Test + hd.Raw_Data_File_Name_Extension
    Raw2DCNNPredict.Run(Test_File,
                        Model_Path,
                        Freqs_Num,
                        hd.Samples_Num,
                        Model_Name)

