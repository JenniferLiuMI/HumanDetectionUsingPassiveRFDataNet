"""
Created by Bing Liu
Export random bands data
"""
import sys
import Header as hd
import ExportBand
import random
import numpy as np
import Common as com

com.Remove_Data_File(hd.Raw_Data_Random_Band, hd.Raw_Data_File_Name_Extension)
freq_full = com.Generate_Feq_List_Ful()
indices = sorted(random.sample(range(1, len(freq_full)), 128))
freq_export = np.take(freq_full, indices, axis = 0).astype(int)
np.savetxt(hd.Root +  'RFRawData\FreqsList\RandomBandFreqs.txt', freq_export, delimiter=',', fmt='%d')
ExportBand.Export(hd.Root + hd.Random_Band_Freq_List_File, hd.Raw_Data_Full_Band, hd.Raw_Data_Random_Band )