"""
Created by Bing Liu
Export inactive bands data
"""

import sys
import Header as hd
import ExportBand
import Common as com

com.Remove_Data_File(hd.Raw_Data_Inactive_Band, hd.Raw_Data_File_Name_Extension)
ExportBand.Export(hd.Root + hd.Inactive_Band_Freq_List_File, hd.Raw_Data_Full_Band, hd.Raw_Data_Inactive_Band )