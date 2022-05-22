"""
Created by Bing Liu
Human detection using IQ of inactive bands excluding cell bands data
"""
import Common as com
import Header as hd
import IQCreateDataSet
import IQCNNTrain
import IQCNNPredict

Model_Name = "CNN"
Create_Dataset = False
Training = False
Predict = True

IQ_Data_File_Path = hd.IQ_Data_File_Path_Inactive_Band_ExcludedCell
Data_Set_Path = hd.IQ_Data_Set_Path_Inactive_Band_ExcludedCell
Model_Path = hd.IQ_Model_Path_Inactive_Band_ExcludedCell

Freqs_Num = len(com.Load_Feq_List(hd.Root + hd.Freq_List_File_Inactive_Band_ExcludedCell ))

if Create_Dataset == True:

    com.Remove_Data_File(Data_Set_Path, hd.Raw_Data_File_Name_Extension)
    IQCreateDataSet.Create(IQ_Data_File_Path,
                              Data_Set_Path,
                              hd.Data_Set_Train,
                              hd.Data_Set_Valid,
                              hd.Data_Set_Test,
                              1500,
                              ["01"],
                              Freqs_Num)

if Training == True:

    EPOCHS = 10
    BATCH_SIZE = 16
    IQCNNTrain.Run(EPOCHS,
                      BATCH_SIZE,
                      Model_Path,
                      Data_Set_Path,
                      hd.Data_Set_Train,
                      hd.Data_Set_Valid,
                      hd.Data_Set_Test,
                      Freqs_Num,
                      hd.Samples_Num/2,
                      Model_Name)

if Predict == True:

    Test_File = Data_Set_Path + hd.Data_Set_Test + hd.Raw_Data_File_Name_Extension
    IQCNNPredict.Run(Test_File,
                        Model_Path,
                        Freqs_Num,
                        hd.Samples_Num/2,
                        Model_Name)