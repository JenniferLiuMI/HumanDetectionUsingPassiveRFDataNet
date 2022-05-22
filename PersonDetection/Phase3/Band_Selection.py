"""
Created by Bing Liu
Band selection using PCA or RFE-LR in phase 3
"""
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import pickle
import warnings
import Com_Power
import Header as hd
import Common as com

#Band selection using PCA
def PCA_Band_Selection(Sample_File_Num, Select_Band_Num):

    freqs, power_Yes, power_No = Get_Sample_File(Sample_File_Num)
    input =  np.concatenate((power_Yes, power_No), axis=0)

    pca = PCA()
    pca.fit(input)
    T = pca.transform(input)
    initial_feature_names = []
    for i in range(len(freqs)):
        initial_feature_names.append(freqs[i])

    Export_Selected_Bands_PCA(Select_Band_Num, freqs, T, pca.components_, initial_feature_names) 

#Export selected bands
def Export_Selected_Bands(File_Name, bands):
    with open(File_Name, 'w') as filehandle:
        for listitem in bands:
            filehandle.write('%s\n' % listitem)

#Export ranks
def Export_Ranks(File_Name, ranks):
    with open(File_Name, 'w') as filehandle:
        for i in range( len(ranks)):
            filehandle.write('%s\n' % ranks[i])

#Export selected bands via PCA
def Export_Selected_Bands_PCA(Select_Band_Num, Freqs, transformed_features, components_, columns):
    num_columns = len(columns)

    xvector = components_[0] 
    yvector = components_[1]
    zvector = components_[2]

    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2+ zvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)

    top_features = []
    sorted_features = np.reshape( important_features, len(important_features)*len(important_features[0]), order='F')[len(important_features):]
    rank = len(sorted_features) - np.reshape(np.array([np.where(sorted_features == x) for x in Freqs])+1, len(sorted_features))
    for i in range(Select_Band_Num):
        freq = important_features[i][1]
        top_features.append(freq)
  
    Export_Ranks(hd.PCA_Rank_List_File.format(Select_Band_Num), rank)
    Export_Selected_Bands(hd.PCA_Selected_Band_List_File.format(Select_Band_Num), top_features)
    Export_Selected_Bands(hd.PCA_Selected_Band_List_File_Sorted.format(Select_Band_Num), top_features)

#Get sample files
def Get_Sample_File(Sample_File_Num):
    file_list_yes, file_list_no = Com_Power.Get_Power_Avg_List(Sample_File_Num)
    power_Yes = None
    power_No = None
    freqs = None
    with open(file_list_yes[0], "rb") as f:
        freq_arr, lables_per_file, freq_powers_per_file = pickle.load(f)
        freqs = freq_arr
        power_Yes = np.zeros((Sample_File_Num, len(freq_powers_per_file)), dtype=np.float)
        power_No = np.zeros((Sample_File_Num, len(freq_powers_per_file)), dtype=np.float)

    i = 0
    for file in file_list_yes:
        with open(file_list_yes[i], "rb") as f:
            freq_arr, lables_per_file, freq_powers_per_file = pickle.load(f)
            power_Yes[i] = freq_powers_per_file
            i = i + 1
    i = 0
    for file in file_list_no:
        with open(file_list_no[i], "rb") as f:
            freq_arr, lables_per_file, freq_powers_per_file = pickle.load(f)
            power_No[i] = freq_powers_per_file
            i = i + 1
    return freqs, power_Yes, power_No

#Band selection using RFE-LR
def RFE_Band_Selection(Sample_File_Num, Select_Band_Num):

    freqs, power_Yes, power_No = Get_Sample_File(Sample_File_Num)
    x =  np.concatenate((power_Yes, power_No), axis=0)

    label_Yes = np.ones(len(power_Yes))
    label_No = np.zeros(len(power_No))

    y = np.concatenate((label_Yes,  label_No), axis=0)
    
    warnings.filterwarnings("ignore")
    model = LogisticRegression()
    rfe = RFE(model, 1)
    fit = rfe.fit(x, y)
    warnings.filterwarnings("always")
    rfe.ranking_
    rank_num = len(rfe.ranking_)
    rank =rank_num -rfe.ranking_
    selected_bands = np.take(freqs, np.argwhere(fit.ranking_ <=Select_Band_Num))
    selected_bands = selected_bands.reshape(len(selected_bands))
    Export_Ranks(hd.RFE_Rank_List_File.format(Select_Band_Num), rank)
    Export_Selected_Bands(hd.RFE_Selected_Band_List_File.format(Select_Band_Num), selected_bands)

#Run band selection using PCA or RFE-LR
#RFE_Band_Selection(600, 80)
PCA_Band_Selection(600, 1446)