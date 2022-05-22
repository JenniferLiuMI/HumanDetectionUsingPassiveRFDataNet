"""
Created by Bing Liu
Human detection using amplitude of selected bands data collected at home and office
"""
import Common as com
import Header as hd
import AMPCreateDataSet
import AMPCNNTrain
import AMPCNNPredict

Model_Name = "CNN"
Freqs_Num = 197
Create_Dataset = True
Training = True
Predict = False

IQ_Data_File_Path = hd.IQ_Data_File_Path_Band_Selective
Data_Set_Path = hd.AMP_Data_Set_Path_Selective_Band
Model_Path = hd.AMP_Model_Path_Selective_Band

if Create_Dataset == True:

    com.Remove_Data_File(Data_Set_Path, hd.Freq_List_File_Random_Band)
    AMPCreateDataSet.Create(IQ_Data_File_Path,
                              Data_Set_Path,
                              hd.Data_Set_Train,
                              hd.Data_Set_Valid,
                              hd.Data_Set_Test,
                              1500,
                              ["01", "05"],
                              Freqs_Num)

if Training == True:

    EPOCHS = 10
    BATCH_SIZE = 16
    AMPCNNTrain.Run(EPOCHS,
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
    AMPCNNPredict.Run(Test_File,
                        Model_Path,
                        Freqs_Num,
                        hd.Samples_Num/2,
                        Model_Name)