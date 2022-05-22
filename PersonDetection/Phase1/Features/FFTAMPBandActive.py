"""
Created by Bing Liu
Human detection using FFT amplitude of active bands data
"""
import Common as com
import Header as hd
import FFTAMPCreateDataSet
import FFTAMPCNNTrain
import FFTAMPCNNPredict

Model_Name = "CNN"
Create_Dataset = False
Training = False
Predict = True

FFT_Data_File_Path = hd.FFT_Data_File_Path_Active_Band
Data_Set_Path = hd.FFT_AMP_Data_Set_Path_Active_Band
Model_Path = hd.FFT_AMP_Model_Path_Active_Band

Freqs_Num = len(com.Load_Feq_List(hd.Root + hd.Freq_List_File_Active_Band ))

if Create_Dataset == True:

    com.Remove_Data_File(Data_Set_Path, hd.Raw_Data_File_Name_Extension)
    FFTAMPCreateDataSet.Create(FFT_Data_File_Path,
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
    FFTAMPCNNTrain.Run(EPOCHS,
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
    FFTAMPCNNPredict.Run(Test_File,
                        Model_Path,
                        Freqs_Num,
                        hd.Samples_Num/2,
                        Model_Name)