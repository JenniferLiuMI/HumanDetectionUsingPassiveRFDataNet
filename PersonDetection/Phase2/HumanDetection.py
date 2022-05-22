"""
Created by Bing Liu
Human detection using bands selected by PCA or RFE-LR
"""
from sklearn.decomposition import IncrementalPCA
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KDTree
from sklearn import neighbors
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
import os
import threading
import pickle
import matplotlib.pyplot as plt
import math
import warnings
import Common as com
import Header as hd

# Confugirations
band_selection_num = 0
band_selection_raw_file_num = 0
location_code_Yes = ""
location_code_No = ""
location_name = ""
location_code_all = ["01", "06", "07", "08", "09", "12", "13"]

Freq_X_Limit_Low = 24
Freq_X_Limit_High = 1760

band = "Full"
yes = "_Yes"
no = "_No"
freqs_all_file_name = hd.Phase2_Root_Path + "/" + 'FreqFullBand.txt'

log_list = []
raw_data_path = hd.Raw_Data_Full_Band

band_selection_power_file_name_Yes = ""
band_selection_power_file_name_No = ""
band_selection_power_avg_file_name_Yes = ""
band_selection_power_avg_file_name_No = ""
selected_bands_list_file = ""
rank_list_file = ""
power_selected_file_Yes = ""
power_selected_file_No = ""
model_file_name = ""
Lable_fontsize=35
legend_fontsize=35
ticks_fontsize=35
Path_Classification = ""

#Initialize the parameters
def init(feature_selection_path,
         band_selection_num_p,
         band_selection_raw_file_num_p,
         location_code_Yes_p,
         location_code_No_p):
    global Path_Classification
    Path_Classification = feature_selection_path

    global band_selection_num
    band_selection_num = band_selection_num_p
    
    global band_selection_raw_file_num
    band_selection_raw_file_num = band_selection_raw_file_num_p
    
    global location_code_Yes
    global location_code_No
    location_code_Yes = location_code_Yes_p
    location_code_No = location_code_No_p
    
    global location_name
    location_name = hd.Colect_Locations.get(location_code_Yes)
    
    global band_selection_power_file_name_Yes
    band_selection_power_file_name_Yes = Path_Classification + location_code_Yes + "/Power_" +band +"_Samples_" + str(band_selection_raw_file_num) + '_' + location_code_Yes + yes + hd.Raw_Data_File_Name_Extension
    
    global band_selection_power_file_name_No
    band_selection_power_file_name_No = Path_Classification + location_code_Yes + "/Power_"  + band +"_Samples_" + str(band_selection_raw_file_num) + '_' + location_code_No + no + hd.Raw_Data_File_Name_Extension
    
    global band_selection_power_avg_file_name_Yes
    band_selection_power_avg_file_name_Yes = Path_Classification + location_code_Yes + "/"  + 'Power_Avg_' + band +"_Samples_" + str(band_selection_raw_file_num) + '_' + location_code_Yes + yes + hd.Text_File_Extension
    
    global band_selection_power_avg_file_name_No
    band_selection_power_avg_file_name_No = Path_Classification + location_code_Yes+ "/"   + 'Power_Avg_' + band +"_Samples_" + str(band_selection_raw_file_num) + '_' + location_code_No + no + hd.Text_File_Extension
    
    global selected_bands_list_file
    selected_bands_list_file = Path_Classification + location_code_Yes+ "/"   + 'Band_Selection_' + 'Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_Yes + hd.Text_File_Extension

    global rank_list_file
    rank_list_file = Path_Classification + location_code_Yes+ "/"   + 'Band_Ranks_' + 'Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_Yes + hd.Text_File_Extension
    
    global power_selected_file_Yes
    power_selected_file_Yes = Path_Classification + location_code_Yes + "/"  + 'Power_Selected' + '_Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_Yes + yes + hd.Raw_Data_File_Name_Extension
    
    global power_selected_file_No
    power_selected_file_No = Path_Classification + location_code_Yes + "/"  + 'Power_Selected' + '_Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_No + no + hd.Raw_Data_File_Name_Extension
    
    global model_file_name
    model_file_name = Path_Classification + location_code_Yes + "/"  + 'Power_Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_Yes

    global power_all_Yes_file_name
    power_all_Yes_file_name = hd.Phase2_Power_Path + 'Power_All_Yes' + '_' + location_code_Yes + hd.Raw_Data_File_Name_Extension

    global power_all_No_file_name
    power_all_No_file_name = hd.Phase2_Power_Path + 'Power_All_No'+ '_' + location_code_No +  hd.Raw_Data_File_Name_Extension

#Defines models of SGD, DT, KNN and SVM
models = [
            (SGDClassifier(loss="hinge", max_iter=1000, tol = None), "SGD"),
            (tree.DecisionTreeClassifier(), "DT"),
            (neighbors.KNeighborsClassifier(9, weights='uniform'), "KNN"),
            (svm.SVC(gamma = 'auto',probability=True), "SVM")
         ]

#Caculate the power in each frequency band
def Get_Freqs_Powers(raw_file_list):

    freqs = []
    power_list_all = []
    for raw_file in raw_file_list:

        with open(raw_file, "rb") as f:
            freqs = pickle.load(f)
            lable_arr = pickle.load(f)
            raw_data_per_file = np.asarray(pickle.load(f))
        raw_data_per_file = raw_data_per_file/127.5 - 1
        power_per_file = np.linalg.norm(raw_data_per_file, axis=1)
        power_per_file = power_per_file*power_per_file
        db = 10*np.log10(power_per_file/len(raw_data_per_file[0])/2)
        power_list_all.append(db)
    power_arry = np.asarray(power_list_all)
    return freqs, power_arry

#Export the the power in each frequency band
def Export_Power_Bands():

    file_name_list_Yes= []
    file_name_list_No= []

    file_name_list_Yes = com.Get_File_name_List_1(raw_data_path, location_code_Yes, str(hd.Person_Present_Yes), band_selection_raw_file_num)
    file_name_list_No = com.Get_File_name_List_1(raw_data_path, location_code_No, str(hd.Person_Present_No), band_selection_raw_file_num)

    freqs, power = Get_Freqs_Powers(file_name_list_Yes)

    export frequency list
    with open(freq_file_name, 'w') as filehandle:
        for listitem in freqs:
            filehandle.write('%s\n' % listitem)

    #export power for each frequency yes
    with open(band_selection_power_file_name_Yes, 'wb') as f_export:
        pickle.dump(power, f_export, pickle.HIGHEST_PROTOCOL)

    #export average power yes
    power_mean = np.mean(power, axis=0)
    with open(band_selection_power_avg_file_name_Yes, 'w') as filehandle:
        for listitem in power_mean:
            filehandle.write('%s\n' % listitem)

    #export power for each frequency no
    freqs, power = Get_Freqs_Powers(file_name_list_No)
    with open(band_selection_power_file_name_No, 'wb') as f_export:
        pickle.dump(power, f_export, pickle.HIGHEST_PROTOCOL)

    #export average power no
    power_mean = np.mean(power, axis=0)
    with open(band_selection_power_avg_file_name_No, 'w') as filehandle:
        for listitem in power_mean:
            filehandle.write('%s\n' % listitem)
    
    print("Export band done")

#Draw the the powers of frequency bands
def Draw_Power_Avg( ):

    Freqs = np.array(com.Import_From_Text_To_Int(freqs_all_file_name))/1000000
    Avg_Power_Full_Home_Yes = com.Import_From_Text(band_selection_power_avg_file_name_Yes)
    Avg_Power_Full_Home_No = com.Import_From_Text(band_selection_power_avg_file_name_No)

    fig = plt.figure()
    plt.title('Average Power of ' + location_name +' Bands: full ' + ' Samples: ' + str(band_selection_raw_file_num) )
    plt.plot( Freqs, Avg_Power_Full_Home_Yes, color='red', linewidth=0.5, label='Occupied')
    plt.plot( Freqs, Avg_Power_Full_Home_No, color='blue', linewidth=0.5, label='Unoccupied')
    plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()   
    plt.savefig(Path_Classification + location_code_Yes + "/"  + 'Power_Avg_' + band +"_Samples_" + str(band_selection_raw_file_num) + '_' + location_code_Yes + hd.JPG_File_Extension, dpi =400)
    plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
    plt.ylim(-45, -5)
    plt.show()
    plt.close()

#Draw the the powers of all frequency bands
def Draw_Power_Avg_All( ):

    Freqs = np.array(com.Import_From_Text_To_Int(freqs_all_file_name))/1000000

    with open(power_all_Yes_file_name, "rb") as f:
        Avg_Power_Yes = pickle.load(f)
        Avg_Power_Yes = np.mean(Avg_Power_Yes, axis=0)

    with open(power_all_No_file_name, "rb") as f:
        Avg_Power_No = pickle.load(f)
        Avg_Power_No = np.mean(Avg_Power_No, axis=0)

    fig = plt.figure()
    plt.title(location_name)
    plt.plot( Freqs, Avg_Power_Yes, color='red', linewidth=0.5, label='Occupied')
    plt.plot( Freqs, Avg_Power_No, color='blue', linewidth=0.5, label='Unoccupied')
    plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
    plt.ylim(-45, -5)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()   
    plt.savefig(Path_Classification  + location_code_Yes+ "/"  + 'Power_Avg_' + location_code_Yes + hd.JPG_File_Extension, dpi =400)
    plt.show()
    plt.close()

#Select frequency bands using PCA
def PCA_Band_Selection(band_selection_raw_file_num_p):

    with open(band_selection_power_file_name_Yes, "rb") as f:
        power_Yes = pickle.load(f)

    with open(band_selection_power_file_name_No, "rb") as f:
        power_No = pickle.load(f)
    input =  np.concatenate((power_Yes, power_No), axis=0)
    pca = PCA()
    pca.fit(input)
    T = pca.transform(input)
    initial_feature_names = []
    freqs = com.Import_From_Text_To_Int(freqs_all_file_name)
    for i in range(len(freqs)):
        initial_feature_names.append(freqs[i])

    #Draw
    colors = ['red', 'blue']
    target_names = ['Occupied', 'Unoccupied']
    label_Yes = np.ones(band_selection_raw_file_num_p)
    label_No = np.zeros(band_selection_raw_file_num_p)
    label = np.concatenate((label_Yes,  label_No), axis=0)
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(T[label == i, 0], T[label == i, 1],
                    color=color, s=200, lw=2, label=target_name)
    
    plt.legend(loc="upper right", shadow=False, scatterpoints=1, fontsize = legend_fontsize, borderaxespad=0.5,)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    fig_name = Path_Classification + location_code_Yes + "/"  + 'PCA_' + 'Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_Yes + hd.JPG_File_Extension
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()   
    plt.savefig(fig_name, dpi =400)
    plt.show()
    plt.close()

    return T, pca.components_, initial_feature_names

#Export the selected frequency bands
def Export_Selected_Bands(bands):
    with open(selected_bands_list_file, 'w') as filehandle:
        for listitem in bands:
            filehandle.write('%s\n' % listitem)

#Draw the the ranks of frequency bands
def Export_Ranks(ranks):
    with open(rank_list_file, 'w') as filehandle:
        for i in range( len(ranks)):
            filehandle.write('%s\n' % ranks[i])

#Export the frequency bands selected by PCA
def Export_Selected_Bands_PCA(transformed_features, components_, columns):
    """
    This function will return the most "important" 
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns)

    xvector = components_[0] 
    yvector = components_[1]
    zvector = components_[2]

    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2+ zvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)

    Freqs = np.array(com.Import_From_Text_To_Int(freqs_all_file_name))
    top_features = []
    sorted_features = np.reshape( important_features, len(important_features)*len(important_features[0]), order='F')[len(important_features):]
    rank = len(sorted_features) - np.reshape(np.array([np.where(sorted_features == x) for x in Freqs])+1, len(sorted_features))
    for i in range(band_selection_num):
        freq = important_features[i][1]
        top_features.append(freq)
    
    with open(selected_bands_list_file, 'w') as filehandle:
        for listitem in top_features:
            filehandle.write('%s\n' % listitem)

    Export_Ranks(rank)
    Export_Selected_Bands(top_features)

#Maxmize the plot window
def maximize(f):
    plot_backend = plt.get_backend()

    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt4Agg':
        mng.window.showMaximized()

    if plot_backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))
    elif plot_backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((0, 0))
    else:
        f.canvas.manager.window.move(0, 0)
#Plot the ranks
def Draw_Ranks():
    rank = com.Import_From_Text_To_Int(rank_list_file)

    Freqs = np.array(com.Import_From_Text_To_Int(freqs_all_file_name))/1000000
    fig = plt.figure()
    plt.bar(Freqs, rank, linewidth = 0.1)
    plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
    plt.ylim(0, max(rank)+10)    
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Rank',fontsize=Lable_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    fig_name = Path_Classification + location_code_Yes + "/"  + 'Band_Rank_' + 'Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_Yes + hd.JPG_File_Extension
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()   
    maximize(fig)
    plt.savefig( fig_name,  bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()

#Plot the selected frequency bands
def Draw_Selected_Bands():

    selected_bands = com.Import_From_Text_To_Int(selected_bands_list_file)
    Freqs = com.Import_From_Text_To_Int(freqs_all_file_name)

    with open(band_selection_power_file_name_Yes, "rb") as f:
        power_all_Yes = pickle.load(f)

    with open(band_selection_power_file_name_No, "rb") as f:
        power_all_No = pickle.load(f)

    power_selected_Yes = np.zeros((len(power_all_Yes), len(selected_bands)), dtype=np.float32)
    power_selected_No = np.zeros((len(power_all_No), len(selected_bands)), dtype=np.float32)

    Avg_Power_Selective_Home_Yes = com.Import_From_Text(band_selection_power_avg_file_name_Yes)
    Avg_Power_Selective_Home_No = com.Import_From_Text(band_selection_power_avg_file_name_No)

    selected_freqs = np.zeros(len(Freqs))
    selected_freqs = selected_freqs * min(Avg_Power_Selective_Home_No)
    count = 0
    for freq in selected_bands:
        index = Freqs.index(freq)
        selected_freqs[index] = Avg_Power_Selective_Home_Yes[index]
        power_selected_Yes[:, count] = power_all_Yes[:, index]
        power_selected_No[:, count] = power_all_No[:, index]
        count = count + 1

    with open(power_selected_file_Yes, 'wb') as f_export:
        pickle.dump(power_selected_Yes, f_export, pickle.HIGHEST_PROTOCOL)

    with open(power_selected_file_No, 'wb') as f_export:
        pickle.dump(power_selected_No, f_export, pickle.HIGHEST_PROTOCOL)

    selected_bands_num =  str(len(selected_bands))
    Freqs = np.array(Freqs)/1000000
    plt.figure()
    plt.title('Band Selection of ' +location_name + ' Samples: ' + str(band_selection_raw_file_num) + ' Selected Bands: ' +selected_bands_num )
    plt.plot(Freqs, selected_freqs, 'o', markersize=8, markerfacecolor='black',markeredgecolor='black',)
    plt.plot( Freqs, Avg_Power_Selective_Home_Yes, color='red', linewidth=1, label='Occupied')
    plt.plot( Freqs, Avg_Power_Selective_Home_No, color='blue', linewidth=1, label='Unoccupied')
    fig_name = Path_Classification + location_code_Yes + "/"  + 'Band_Selection_' + 'Samples_' + str(band_selection_raw_file_num) + '_Bands_' + selected_bands_num + '_' + location_code_Yes + hd.JPG_File_Extension
    plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlim(Freq_X_Limit_Low, Freq_X_Limit_High)
    plt.ylim(-45, -5)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()   
    plt.show()
    plt.savefig(fig_name, dpi =400)
    plt.close()

#Human detection using bands data selected by PCA
def PCA_Classification():

    title = 'PCA ' + str(band_selection_raw_file_num) + ' Samples ' + str(band_selection_num) + ' Bands'
    
    with open(power_selected_file_Yes, "rb") as f:
        power_Yes = pickle.load(f)

    with open(power_selected_file_No, "rb") as f:
        power_No = pickle.load(f)

    label_Yes = np.ones(len(power_Yes))
    label_No = np.zeros(len(power_No))

    X = np.concatenate((power_Yes,  power_No), axis=0)
    y = np.concatenate((label_Yes,  label_No), axis=0)

    n_components = 2

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    colors = ['red', 'blue']
    target_names = ['Occupied', 'Unoccupied']

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    color=color, lw=2, label=target_name)

    plt.title( title)
    plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize = legend_fontsize)
    fig_name = Path_Classification + location_code_Yes + "/"  + 'PCA_' + 'Samples_' + str(band_selection_raw_file_num) + '_Bands_' + str(band_selection_num) + '_' + location_code_Yes + hd.JPG_File_Extension
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()   
    plt.savefig(fig_name, dpi =400)
    plt.show()
    plt.close()

#Split the data into training and test set
def split_data(x,y, num, percent):

    train_num = (int)(num*percent)
    test_num = num-train_num
    x, y = shuffle(x, y)

    index_Yes = (np.argwhere(y > 0))
    index_Yes = np.reshape(index_Yes, len(index_Yes))

    index_No = (np.argwhere(y < 1))
    index_No = np.reshape(index_No, len(index_No))

    x_train_Yes =  np.take(x, index_Yes[:train_num], axis=0)
    y_train_Yes =  np.take(y, index_Yes[:train_num])

    x_train_No =  np.take(x, index_No[:train_num], axis=0)
    y_train_No =  np.take(y, index_No[:train_num])
    
    x_test_Yes =  np.take(x, index_Yes[train_num:], axis=0)
    y_test_Yes =  np.take(y, index_Yes[train_num:])

    x_test_No =  np.take(x, index_No[train_num:], axis=0)
    y_test_No =  np.take(y, index_No[train_num:])

    x_train = np.concatenate((x_train_Yes,  x_train_No), axis=0)
    y_train = np.concatenate((y_train_Yes,  y_train_No), axis=0)
    x_train, y_train = shuffle(x_train, y_train)

    x_test = np.concatenate((x_test_Yes,  x_test_No), axis=0)
    y_test = np.concatenate((y_test_Yes,  y_test_No), axis=0)
    x_test, y_test = shuffle(x_test, y_test)

    return x_train,x_test,y_train,y_test

#Export powers of all data
def export_all_power():

    file_num = 150

    file_name_list_Yes = com.Get_File_name_List_1(raw_data_path, location_code_Yes, str(hd.Person_Present_Yes), file_num)
    file_name_list_No = com.Get_File_name_List_1(raw_data_path, location_code_No, str(hd.Person_Present_No), file_num)

    freqs_all, power_all_Yes = Get_Freqs_Powers(file_name_list_Yes)
    freqs_all, power_all_No = Get_Freqs_Powers(file_name_list_No)

    with open(power_all_Yes_file_name, 'wb') as f_export:
        pickle.dump(power_all_Yes, f_export, pickle.HIGHEST_PROTOCOL)

    with open(power_all_No_file_name, 'wb') as f_export:
        pickle.dump(power_all_No, f_export, pickle.HIGHEST_PROTOCOL)

#Plot ROC
def plot_ROC_Figure(fpr, tpr, roc_auc, name):
    plt.figure()
    lw = 5
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of ' + name)
    plt.legend(loc="lower right")
    plt.show()

#Generate ROC
def get_ROC(x_train, y_train, x_test, y_test, model, name, classifier=None):

    if name == "KNN" or name == "DT" or name =="SVM":
        y_scores = classifier.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        fpr, tpr, roc_auc
        plot_ROC_Figure(fpr, tpr, roc_auc, name)
        return {'classifier':name,
                'fpr':fpr, 
                'tpr':tpr, 
                'auc':roc_auc}
    else:
        y_score = classifier.decision_function(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plot_ROC_Figure(fpr, tpr, roc_auc, name)
        return {'classifier':name,
                'fpr':fpr, 
                'tpr':tpr, 
                'auc':roc_auc}
#Plot ROC
def plot_ROC(roc_table):
    fig = plt.figure(figsize=(8,6))

    for i in range(len (roc_table)):
        name  = roc_table[i].get('classifier')
        line_color = 'black'
        if name == "SGD":
            line_color = 'red'
        elif name == "KNN":
            line_color = 'goldenrod'
        elif name == "SVM":
            line_color = 'brown'
        elif name == "DT":
            line_color = 'green'
        
        if name == "DT":
            plt.plot(roc_table[i].get('fpr'), 
                     roc_table[i].get('tpr'),
                     color = line_color,
                     label="{},   AUC={:.3f}".format(name, roc_table[i].get('auc')))
        else:
            plt.plot(roc_table[i].get('fpr'), 
                     roc_table[i].get('tpr'), 
                     color = line_color,
                     label="{}, AUC={:.3f}".format(name, roc_table[i].get('auc')))
    # Area Under the Curve
    plt.plot([0,1], [0,1], color='gray', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.savefig("a.jpg")
    plt.show()

#Train the models
def Train(percentage = 0.6, run_training_sample_num=False):

    with open(power_all_Yes_file_name, "rb") as f:
        power_all_Yes = pickle.load(f)

    with open(power_all_No_file_name, "rb") as f:
        power_all_No = pickle.load(f)

    selected_bands = com.Import_From_Text_To_Int(selected_bands_list_file)

    power_selected_Yes = np.zeros((len(power_all_Yes), len(selected_bands)), dtype=np.float32)
    power_selected_No = np.zeros((len(power_all_No), len(selected_bands)), dtype=np.float32)

    freqs_all = com.Import_From_Text_To_Int(freqs_all_file_name)
    count = 0
    for freq in selected_bands:
        index = freqs_all.index(freq)
        power_selected_Yes[:, count] = power_all_Yes[:, index]
        power_selected_No[:, count] = power_all_No[:, index]
        count = count + 1


    label_Yes = np.ones(len(power_selected_Yes))
    label_No = np.zeros(len(power_selected_No))

    x = np.concatenate((power_selected_Yes,  power_selected_No), axis=0)
    y = np.concatenate((label_Yes,  label_No), axis=0)

    x_train,x_test,y_train,y_test = split_data(x,y,len(label_Yes),percentage)

    warnings.filterwarnings("ignore")
    if (run_training_sample_num == True):
        temp =str(location_code_Yes) + "," + str(band_selection_raw_file_num) + ',' + str(band_selection_num) + ',' + str(int(percentage*len(x))) 
    else:
        temp =str(location_code_Yes) + "," + str(band_selection_raw_file_num) + ',' + str(band_selection_num)
    print(temp)
    print("{:<20}  {:>6}".format ("Classifier", ":  TP TN FP FN Accuracy"))
    log = temp
    ROC_table = []

    for (model, name) in models:
        classifier=model
        classifier.fit(x_train,y_train)
        predictions=classifier.predict(x_test)
        accuracy = "{:.2f}%".format(accuracy_score(y_test,predictions)*100.0)
        TP = np.count_nonzero(predictions * y_test)
        TN = np.count_nonzero((predictions - 1) * (y_test - 1))
        FP = np.count_nonzero(predictions * (y_test - 1))
        FN = np.count_nonzero((predictions - 1) * y_test)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_value = 2 * precision * recall / (precision + recall)
        precision = "{:.2f}%".format(precision*100.0)
        recall = "{:.2f}%".format(recall*100.0)
        f1_value = "{:.2f}%".format(f1_value*100.0)

        F1 = "," + str(TP) + "," + str(TN) + ","  + str(FP) + "," + str(FN)

        print("{:<20}  {:>6}".format (name, ": " + F1.replace(",", " ")+ "   "+ accuracy))
        log = log + F1 + "," + precision + "," + recall+ "," + f1_value + "," + accuracy
        ROC = get_ROC(x_train, y_train, x_test, y_test, model,name, classifier)
        ROC_table.append( ROC)
        dump_file_name = model_file_name + '_' + name + hd.Model_File_Extension
        with open(dump_file_name, 'wb') as f_export:
            pickle.dump(classifier, f_export, pickle.HIGHEST_PROTOCOL)
    log_list.append(log)
    plot_ROC(ROC_table)

    warnings.filterwarnings("always")
  
#Setup the paparmeters
def setup(feature_selection_path, location_code_Yes, band_selection_raw_file_num_p, band_selection_num_p):
    if location_code_Yes == "08" or location_code_Yes == "09":
        init(feature_selection_path, band_selection_num_p, band_selection_raw_file_num_p, location_code_Yes, "06")
    elif location_code_Yes == "12":
        init(feature_selection_path, band_selection_num_p, band_selection_raw_file_num_p, location_code_Yes, "01")
    elif location_code_Yes == "13":
        init(feature_selection_path, band_selection_num_p, band_selection_raw_file_num_p, location_code_Yes, "07")
    else:
        init(feature_selection_path, band_selection_num_p, band_selection_raw_file_num_p, location_code_Yes, location_code_Yes)

#Train the models
def run_training_sample_num(feature_selection_path, file_name):
    sample_num = 30
    band_selection_num = 20
    percentage_step = 0.1

    for location_code_Yes in ["01", "06", "07", "08", "09", "12", "13"]:
        log_list.clear()
        setup(feature_selection_path, location_code_Yes, sample_num, band_selection_num)
        for i in range(1,9,1):
            Train(i*percentage_step, True)
        detection_log_txt = hd.Phase2_Root_Path+ "/"  + file_name + hd.CSV_File_Extension
        with open(detection_log_txt, 'a+') as filehandle:
            for listitem in log_list:
                filehandle.write('%s\n' % listitem)

#Train all models
def run_training_all(feature_selection_path):
    sample_num_step = 10
    band_num_step = 10

    for location_code_Yes in ["01", "06", "07", "08", "09", "12", "13"]:
        log_list.clear()
        for i in range(0,6,1):
            band_selection_raw_file_num_p = sample_num_step + sample_num_step*i
            for j in range(0,15,1):
                band_selection_num_p = band_num_step + band_num_step*j
                setup(feature_selection_path, location_code_Yes, band_selection_raw_file_num_p, band_selection_num_p)
                Train()
        detection_log_txt = hd.Phase2_Root_Path+ "/"  + 'Detection_Result_F1' + hd.CSV_File_Extension
        with open(detection_log_txt, 'a+') as filehandle:
            for listitem in log_list:
                filehandle.write('%s\n' % listitem)

                print( "Location:" + location_code_Yes + ' Samples:' + str(band_selection_raw_file_num) + ' Bands:' + str(band_selection_num))
                Train()
                print("---------------------------------")

#Band selection using PCA
def PCA_run_band_selection(feature_selection_path):
    #01	StRmP1
    #12	StRmP2
    #07	BdRmP1
    #13	BdRmP2
    #08	CrP1
    #06	CrP2
    #09	CrP3
    location_code_Yes = "12"
    log_list.clear()
    band_selection_raw_file_num_p = 60
    band_selection_num_p = 30
    setup(feature_selection_path, location_code_Yes, band_selection_raw_file_num_p, band_selection_num_p)
    Export_Power_Bands()
    Draw_Power_Avg()
    T, components, columns = PCA_Band_Selection(band_selection_raw_file_num_p)
    Export_Selected_Bands_PCA(T, components, columns)
    Draw_Ranks()
    Draw_Selected_Bands()

#Band selection using RFE-LR
def RFE_run_band_selection(feature_selection_path):
    location_code_Yes = "12"
    log_list.clear()
    band_selection_raw_file_num_p = 60
    band_selection_num_p = 30
    setup(feature_selection_path, location_code_Yes, band_selection_raw_file_num_p, band_selection_num_p)
    RFE_Band_Selection(band_selection_num)
    Draw_Ranks()
    Draw_Selected_Bands()

#Execute end to end using PCA
def run_PCA():    
    #for location_code_Yes in location_code_all:
    #for location_code_Yes in  ["12", "13","09"]:
    location_code_Yes = "01"
    log_list.clear()
    band_selection_raw_file_num_p = 40
    band_selection_num = 40
    setup(hd.PCA_Path_Classification, location_code_Yes, band_selection_raw_file_num_p, band_selection_num)
    print("Export power...")
    Export_Power_Bands()
    Draw_Power_Avg()
    print("Selecting...")
    T, components, columns = PCA_Band_Selection(band_selection_raw_file_num_p)
    Export_Selected_Bands_PCA(T, components, columns)
    Draw_Ranks()
    Draw_Selected_Bands()
    print( "Location:" + location_code_Yes + ' Samples:' + str(band_selection_raw_file_num) + ' Bands:' + str(band_selection_num))
    print("Training...")
    Train()
    print("---------------------------------")

    detection_log_txt = hd.Phase2_Root_Path+  'Detection_Result_F1_RFE' + hd.CSV_File_Extension
    with open(detection_log_txt, 'a+') as filehandle:
        for listitem in log_list:
            filehandle.write('%s\n' % listitem)

#Human detection though PCA
def PCA_Run_All():    
    sample_num_step = 10
    band_num_step = 10

    #"01": "**Home1 Antenna postion1",
    #"06": "**Car1 Passenger Seat Left",
    #"07": "**Home2 Huaizheng",
    #"08": "*Car1 Driver Seat",
    #"09": "*Car1 Passenger Seat Right",
    #"12": "*Home1 postion2 far",
    #"13": "*Home2 postion2 far",

    #Uncomment below code which you want to run
    #for location_code_Yes in ["01", "06", "07", "08", "09", "12", "13"]:
    #for location_code_Yes in ["01", "13", "08"]:
    #for location_code_Yes in ["12", "13", "09"]:
    for location_code_Yes in ["08"]:
        for i in range(1,6,1):
            band_selection_raw_file_num_p = sample_num_step + sample_num_step*i
            for j in range(1,15,1):
                band_selection_num_p = band_num_step + band_num_step*j
                setup(hd.PCA_Path_Classification, location_code_Yes, band_selection_raw_file_num_p, band_selection_num_p)

                if i ==1 and j == 0:
                    export_all_power()
                Draw_Power_Avg_All()    

                if j ==0 or j ==1:
                    Export_Power_Bands()
                
                Draw_Power_Avg()
                T, components, columns = PCA_Band_Selection(band_selection_raw_file_num_p)
                Export_Selected_Bands_PCA(T, components, columns)

                Draw_Selected_Bands()
                print( "Location:" + location_code_Yes + ' Samples:' + str(band_selection_raw_file_num) + ' Bands:' + str(band_selection_num))
                Train()
                print("---------------------------------")

        detection_log_txt = hd.Phase2_Root_Path+ "/"  + 'Detection_Result_F1' + hd.CSV_File_Extension
        with open(detection_log_txt, 'a+') as filehandle:
            for listitem in log_list:
                filehandle.write('%s\n' % listitem)

#Bands selection using RFE-LR
def RFE_Band_Selection(Select_Band_Num):

    Avg_Power_Selective_Home_Yes = com.Import_From_Text(band_selection_power_avg_file_name_Yes)
    Avg_Power_Selective_Home_No = com.Import_From_Text(band_selection_power_avg_file_name_No)

    with open(band_selection_power_file_name_Yes, "rb") as f:
        power_Yes = pickle.load(f)

    with open(band_selection_power_file_name_No, "rb") as f:
        power_No = pickle.load(f)
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
    freqs = com.Import_From_Text_To_Int(freqs_all_file_name)
    selected_bands = np.take(freqs, np.argwhere(fit.ranking_ <=Select_Band_Num))
    selected_bands = selected_bands.reshape(len(selected_bands))
    Export_Ranks(rank)
    Export_Selected_Bands(selected_bands)

#Train the models using bands data selected by RFE-LR
def run_RFE_Trian():    
    sample_num_step = 10
    band_num_step = 10

    for location_code_Yes in location_code_all:
        log_list.clear()
        for i in range(0,6,1):
            band_selection_raw_file_num_p = sample_num_step + sample_num_step*i
            for j in range(0,15,1):
                band_selection_num = band_num_step + band_num_step*j
                setup(hd.RFE_Path_Classification, location_code_Yes, band_selection_raw_file_num_p, band_selection_num)
                print( "Location:" + location_code_Yes + ' Samples:' + str(band_selection_raw_file_num) + ' Bands:' + str(band_selection_num))
                Train()
                print("---------------------------------")

        detection_log_txt = hd.Phase2_Root_Path+  'Detection_Result_F1_RFE' + hd.CSV_File_Extension
        with open(detection_log_txt, 'a+') as filehandle:
            for listitem in log_list:
                filehandle.write('%s\n' % listitem)

#Human detection through RFE-LR
def run_RFE():    
    #Uncomment below code which you want to run
    #for location_code_Yes in location_code_all:
    #for location_code_Yes in  ["12", "13","09"]:
    location_code_Yes = "01"
    log_list.clear()
    band_selection_raw_file_num_p = 40
    band_selection_num = 40
    setup(hd.RFE_Path_Classification, location_code_Yes, band_selection_raw_file_num_p, band_selection_num)
    print("Export power...")
    Export_Power_Bands()
    Draw_Power_Avg()
    print("Selecting...")
    RFE_Band_Selection(band_selection_num)
    Draw_Ranks()
    Draw_Selected_Bands()
    print( "Location:" + location_code_Yes + ' Samples:' + str(band_selection_raw_file_num) + ' Bands:' + str(band_selection_num))
    print("Training...")
    Train()
    print("---------------------------------")

    detection_log_txt = hd.Phase2_Root_Path+  'Detection_Result_F1_RFE' + hd.CSV_File_Extension
    with open(detection_log_txt, 'a+') as filehandle:
        for listitem in log_list:
            filehandle.write('%s\n' % listitem)

#Human detection through RFE-LR
def run_RFE_All():    
    sample_num_step = 10
    band_num_step = 10

    #Uncomment below code which you want to run
    #for location_code_Yes in location_code_all:
    #for location_code_Yes in  ["12", "13","09"]:
    for location_code_Yes in  [ "09"]:
        log_list.clear()
        for i in range(1,6,1):
            band_selection_raw_file_num_p = sample_num_step + sample_num_step*i
            for j in range(1,15,1):
                band_selection_num = band_num_step + band_num_step*j
                setup(hd.RFE_Path_Classification, location_code_Yes, band_selection_raw_file_num_p, band_selection_num)
                if j ==0:
                    Export_Power_Bands()
                    Draw_Power_Avg()
                RFE_Band_Selection(band_selection_num)
                Draw_Ranks()
                Draw_Selected_Bands()

                print( "Location:" + location_code_Yes + ' Samples:' + str(band_selection_raw_file_num) + ' Bands:' + str(band_selection_num))
                Train()
                print("---------------------------------")

        detection_log_txt = hd.Phase2_Root_Path+  'Detection_Result_F1_RFE' + hd.CSV_File_Extension
        with open(detection_log_txt, 'a+') as filehandle:
            for listitem in log_list:
                filehandle.write('%s\n' % listitem)

if __name__ == '__main__':
    #Uncomment below code which you want to run
    #PCA_Run_All()

    #PCA
    #run_training_sample_num(PCA_Path_Classification, "Detection_Result_F1SampleNum_PCA")
    #run_training_all(PCA_Path_Classification)
    #PCA_run_band_selection(hd.PCA_Path_Classification)

    #RFE
    #run_RFE_All()
    #run_RFE_Trian()
    #run_training_sample_num(hd.RFE_Path_Classification, "Detection_Result_F1SampleNum_RFE")
    RFE_run_band_selection(hd.RFE_Path_Classification)

    #run_PCA()
    #run_RFE()

