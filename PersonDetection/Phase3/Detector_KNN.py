"""
Created by Bing Liu
Build KNN to detect person in phase 3
"""
from sklearn.decomposition import IncrementalPCA
import numpy as np
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KDTree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sklearn.preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from scipy import interp
import matplotlib.pyplot as plt
import sys
import os
import pickle
import Com_GAN as com_GAN
import Com_Power as com_Power
import pandas as pd
sys.path.insert(1, '/media/win/Code/RFResearch/PersonDetection')

import matplotlib.pyplot as plt
import math
import warnings
import time
import Header as hd
import Common as com
from enum import Enum
import Detector_CNN  as Detector_CNN


#https://www.datacamp.com/community/tutorials/feature-selection-python

class_num = 2
band_selection_num = 0
band_selection_raw_file_num = 0
location_code_Yes = ""
location_code_No = ""
train_location = "01"
#test_location = "01"
#test_location = "05"
#test_location = "07"
test_location = "06"

occupied_color='red'
unoccupied_color='royalblue'
gene_color='gray'

Freq_X_Limit_Low = 24
Freq_X_Limit_High = 1760

band = "Full"
yes = "_Yes"
no = "_No"

log_list = []

class Detection(Enum):
  Train_Gene_Test_Gene = 1
  Train_Real_Test_Gene = 2
  Train_Real_Test_Real = 3

model_file_name = ""
Lable_fontsize=10
legend_fontsize=10
ticks_fontsize=10
Path_Classification = ""
root = '/media/win/Code/RFResearch/GANout/Power_Avg_784_1_3_1_3_1_5_1_2_1_3_1_3_1_5_0_0_Model_Loc_1'
gene_data_dir  = root + '/generated_data'
gene_data_file = root + '/generated_data/gene_power_{}.pkl'.format(test_location)

real_data_dir  = root + '/real_data'
real_data_file_test = root + '/real_data/real_{}.pkl'.format(test_location)
loss_data_file_name = root + '/loss/loss.txt'
loss_figure_file_name = root + '/loss/loss_1.png'
CNN_Folder = gene_data_dir + '/CNN'

models = [
            #(SGDClassifier(loss="hinge", penalty='l2', max_iter=4000), "SGD"),
            #(tree.DecisionTreeClassifier(), "DT"),
            (neighbors.KNeighborsClassifier(9, weights='uniform'), "KNN"),
            #(svm.SVC(C=1.0, coef0=0.0, tol=1e-4, gamma = 'auto',probability=True), "SVM")
         ]

def gene_loss_figure():
    if os.path.exists(loss_data_file_name):
      with open(loss_data_file_name, 'r') as f:
        loss_D, loss_G = com_GAN.get_loss(f)
    com_GAN.draw_loss(loss_figure_file_name, loss_D, loss_G)

def To_One_Hot(y):
    y = y.astype(np.int)
    y_temp = np.zeros((len(y), 2), dtype=np.int)
    for i in range(len(y)):
        y_temp[i, y[i]] = 1
    return y_temp

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

def get_ROC(x_train, y_train, x_test, y_test, model=None, name='', classifier=None):

    if name == "KNN" or name == "DT" or name =="SVM":
        y_scores = classifier.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        fpr, tpr, roc_auc
        #plot_ROC_Figure(fpr, tpr, roc_auc, name)
        return {'classifier':name,
                'fpr':fpr, 
                'tpr':tpr, 
                'auc':roc_auc}
    else:
        y_score = classifier.decision_function(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        #plot_ROC_Figure(fpr, tpr, roc_auc, name)
        return {'classifier':name,
                'fpr':fpr, 
                'tpr':tpr, 
                'auc':roc_auc}

def plot_ROC(roc_table, detection):
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

    #plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    if detection == Detection.Train_Real_Test_Gene:
        plt.savefig( os.path.join(gene_data_dir, "gene_ROC_{}.jpg".format(test_location)),  dpi =400)
    elif detection == Detection.Train_Real_Test_Real:
        plt.savefig( os.path.join(real_data_dir, "real_ROC_{}.jpg".format(test_location)), dpi =400)
    plt.show()

def Draw_One_Power(data_file, location):
    with open(data_file, "rb") as f:
        freqs, x_real, y_real = pickle.load(f)
        if abs(np.mean(x_real)) < 1:
            x_real = x_real*-40

    x_real_No = x_real[np.where(y_real==hd.Person_Present_No)[0]]
    x_real_Yes = x_real[np.where(y_real==hd.Person_Present_Yes)[0]]

    fig_file_name = os.path.join(real_data_dir, "Signle_Occupied_{}_{}".format(location, hd.JPG_File_Extension))
    com_Power.Draw_Freqs_Figure_One(fig_file_name, freqs, x_real_Yes[0].reshape(len(freqs)), line_color=occupied_color,my_figsize=(8, 4))

    fig_file_name = os.path.join(real_data_dir, "Signle_Unoccupied_{}_{}".format(location, hd.JPG_File_Extension))
    com_Power.Draw_Freqs_Figure_One(fig_file_name, freqs, x_real_No[0].reshape(len(freqs)), line_color=unoccupied_color,my_figsize=(8, 4))

def Draw_Average_Power(data_file, is_gene_data=True):
    with open(data_file, "rb") as f:
        freqs, x, y = pickle.load(f)

    if is_gene_data == True:
        fig_file_name = os.path.join(gene_data_dir, "gene_average_power_{}".format(test_location) + hd.JPG_File_Extension)
    else:
        fig_file_name = os.path.join(real_data_dir, "real_average_power_{}".format(test_location) + hd.JPG_File_Extension)

    Plot_Average_Power(freqs, x,y, fig_file_name)

def Draw_Average_Power_gene_real(gene_data_file, real_data_file):

    with open(gene_data_file, "rb") as f:
        freqs, x_gene, y_gene = pickle.load(f)
        if abs(np.mean(x_gene)) < 1:
            x_gene = x_gene*-40

    with open(real_data_file, "rb") as f:
        freqs, x_real, y_real = pickle.load(f)
        if abs(np.mean(x_real)) < 1:
            x_real = x_real*-40

    x_real_No = x_real[np.where(y_real==hd.Person_Present_No)[0]]
    x_real_Yes = x_real[np.where(y_real==hd.Person_Present_Yes)[0]]

    Plot_Average_Power_3(freqs, x_real_Yes, x_real_No,x_gene)

def Plot_Average_Power_3(freqs, x_real_Yes,x_real_No, x_gene):


    x_real_Yes_mean = np.mean(x_real_Yes, axis=0)
    x_real_No_mean = np.mean(x_real_No, axis=0)
    x_gene_Yes_mean = np.mean(x_gene, axis=0)
    fig = plt.figure()
    line_color = 'red'
    plt.plot( freqs, x_real_Yes_mean, linewidth=0.5, color='red', label='Real Occupied')
    plt.plot( freqs, x_real_No_mean, linewidth=0.5, color='royalblue', label='Real Unoccupied')
    plt.plot( freqs, x_gene_Yes_mean, color='gray', linewidth=0.5,  label='Synthesized Occupied')
    plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    fig_file_name = os.path.join(gene_data_dir, "gene_average_power_{}".format(test_location) + hd.JPG_File_Extension)
    plt.savefig(fig_file_name, dpi =400)        
    print('Saved to : {}'.format(fig_file_name))
    plt.close()

    df = pd.DataFrame({'Synthesized': x_gene_Yes_mean})
    df['Occupied'] =  x_real_Yes_mean # positively correlated with 'a'
    df['Unoccupied'] = x_real_No_mean # negatively correlated with 'a'
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.matshow(df.corr(),  interpolation='nearest')
    plt.xticks(range(len(df.columns)), df.columns, fontsize = 8)
    plt.yticks(range(len(df.columns)), df.columns, fontsize = 7)
    #cb = plt.colorbar()
    #cb.ax.tick_params(labelsize=8)

    for (i, j), z in np.ndenumerate(corr):
        #ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        #plt.text(i, j, '{:0.1f}'.format(z), ha='center', va='center',
        #    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))        
        plt.text(i, j, f"{z:.2f}", va="center", ha="center",bbox=dict( boxstyle='round', facecolor='white', edgecolor='0.9'))
    #plt.colorbar()
    fig_file_name = os.path.join(gene_data_dir, "gene_correlations_{}".format(test_location) + hd.JPG_File_Extension)
    plt.savefig(fig_file_name, dpi =400)        
    c1 = np.corrcoef(x_gene_Yes_mean, x_real_Yes_mean)
    print('correlations: {}/{}'.format('synthesized','Occupied'))
    print(c1)
    c2 =np.corrcoef(x_gene_Yes_mean, x_real_No_mean)
    print('correlations: {}/{}'.format('synthesized','Unoccupied'))
    print(c2)
    c3 =np.corrcoef(x_real_Yes_mean, x_real_No_mean)
    print('correlations: {}/{}'.format('Occupied','Unoccupied'))
    print(c3)
    a=0

def Plot_Average_Power(freqs, x,y, fig_file_name):

    if abs(np.mean(x)) < 1:
        x = x*-40
    x_Yes = x[np.where(y==1)[0]]
    x_No = x[np.where(y==0)[0]]
    x_Yes_mean = np.mean(x_Yes, axis=0)
    x_No_mean = np.mean(x_No, axis=0)
    fig = plt.figure()
    line_color = 'red'
    plt.plot( freqs, x_Yes_mean, linewidth=0.5, color='red')
    plt.plot( freqs, x_No_mean, linewidth=0.5, color='royalblue')
    plt.legend(loc="best", borderaxespad=0.5, fontsize = legend_fontsize)
    plt.xlabel('Frequency Band (MHz)',fontsize=Lable_fontsize)
    plt.ylabel('Power (DB)',fontsize=Lable_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    
    plt.savefig(fig_file_name, dpi =400) 
    print('Saved to {}'.format(fig_file_name))
    plt.close()

def Train(train_data_file, test_data_file=None, detection = Detection.Train_Gene_Test_Gene):

    x_train, y_train, x_test, y_test = None, None, None, None,

    with open(train_data_file, "rb") as f:
        freqs, x, y = pickle.load(f)
        if abs(np.mean(x))> 1:    
            x=(1-(x/-40))

    s = np.arange(len(x))
    np.random.shuffle(s)
    x = x[s]
    y = y[s]

    if test_data_file is None:
        trian_sample_num = int(len(x)/2 * 0.7)
        test_sample_num = len(x) - trian_sample_num

        power_Yes_train = x[np.where(y==1)[0]][:trian_sample_num]
        power_No_train = x[np.where(y==0)[0]][:trian_sample_num]

        label_Yes_train = np.ones(trian_sample_num, dtype=np.int)
        label_No_train = np.zeros(trian_sample_num, dtype=np.int)

        x_train = np.concatenate((power_Yes_train, power_No_train), axis=0)
        y_train = np.concatenate( (label_Yes_train, label_No_train), axis=0)
        s = np.arange(len(x_train))
        np.random.shuffle(s)
        x_train = x_train[s]
        y_train = y_train[s]

        power_Yes_test = x[np.where(y==1)[0]][trian_sample_num:]
        power_No_test = x[np.where(y==0)[0]][trian_sample_num:]

        label_Yes_test = np.ones(len(power_Yes_test), dtype=np.int)
        label_No_test = np.zeros(len(power_No_test), dtype=np.int)

        x_test = np.concatenate((power_Yes_test, power_No_test), axis=0)
        y_test = np.concatenate( (label_Yes_test, label_No_test), axis=0)
        s = np.arange(len(x_test))
        np.random.shuffle(s)
        x_test = x_test[s]
        y_test = y_test[s]
    else:
        x_real_No = x[np.where(y==hd.Person_Present_No)[0]]
        y_real_No = np.zeros(len(x_real_No), dtype=np.int)
        x_real_Yes = x[np.where(y==hd.Person_Present_Yes)[0]]
        y_real_Yes = np.ones(len(x_real_Yes), dtype=np.int)

        x_real_No_num_train = int(len(x_real_No)*0.7)
        x_real_Yes_num_train = int(len(x_real_Yes)*0.7)

        x_train = np.concatenate((x_real_No[:x_real_No_num_train],x_real_Yes[:x_real_Yes_num_train]), axis=0)
        y_train = np.concatenate((np.zeros(x_real_No_num_train, dtype=np.int),np.ones(x_real_Yes_num_train, dtype=np.int)), axis=0 )

        with open(test_data_file, "rb") as f:
            freqs, x_test, y_test = pickle.load(f)
            if abs(np.mean(x_test))> 1:    
                x_test=(1-(x_test/-40))
                
            x_test = np.concatenate((x_test, x_real_No[x_real_No_num_train:]), axis=0)
            y_test = np.concatenate((y_test, y_real_No[x_real_No_num_train:]), axis=0)
        s = np.arange(len(x_test))
        np.random.shuffle(s)
        x_test = x_test[s]
        y_test = y_test[s]

        s = np.arange(len(x_train))
        np.random.shuffle(s)
        x_train = x_train[s]
        y_train = y_train[s]


    warnings.filterwarnings("ignore")

    ROC_table = []
    log_list=[]
    print("{:<12}{}".format ("Classifier", ":    TP     TN     FP    FN    Accuracy"))
    for (model, name) in models:
        classifier=model
        if len(np.unique(y_train)) <2:
            print('Only one class')
            break
        classifier.fit(x_train,y_train)
        predictions=classifier.predict(x_test)
        prdeict_Yes = 0
        prdeict_No = 0
        if len(np.where(predictions==1)[0])>0:
            prdeict_Yes = len(np.where(predictions==1)[0])

        if len(np.where(predictions==0)[0]) >0: 
            prdeict_No = len(np.where(predictions==0)[0])
        accuracy = "{:.2f}%".format(accuracy_score(y_test,predictions)*100.0)
        TP = np.count_nonzero(predictions * y_test)
        TN = np.count_nonzero((predictions - 1) * (y_test - 1))
        FP = np.count_nonzero(predictions * (y_test - 1))
        FN = np.count_nonzero((predictions - 1) * y_test)

        com.Log_Setup(CNN_Folder + '/log/', "test")
        com.Log_Info('----------------------------------------------------')
        print_str = '{} Model is trained with: {} Testing on: {}'.format(name, train_location, test_location)
        print(print_str)
        com.Log_Info(print_str)

        print_str = 'Train data: {} real occupied: {} real unoccupied: {}\nTest data: {} gene occupied: {} real unoccupied: {}\n'.format(len(x_train),
                        len(np.where(y_train==hd.Person_Present_Yes)[0]),
                        len(np.where(y_train==hd.Person_Present_No)[0]),
                        len(x_test),
                        len(np.where(y_test==hd.Person_Present_Yes)[0]),
                        len(np.where(y_test==hd.Person_Present_No)[0])
                        )
        print(print_str)
        com.Log_Info(print_str)
        print_str = "{:<12}{}".format ("Model Name", ":    TP     TN     FP    FN    Precision    Recall    F1        Accuracy")
        print(print_str)
        com.Log_Info(print_str)

        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        recall = TP / (TP + FN)

        if precision + recall>0:
            f1_value = 2 * precision * recall / (precision + recall)
        else:
            f1_value = 0
        precision = "{:.2f}".format(precision*100.0)
        recall = "{:.2f}".format(recall*100.0)
        f1_value = "{:.2f}".format(f1_value*100.0)

        print_str = '{:<12}:  {:>5}  {:>5} {:>5} {:>5}     {:>6}     {:>6}     {:>6}     {:>6}'.format(name, TP, TN, FP, FN, precision, recall, f1_value, accuracy)
        print(print_str)
        com.Log_Info(print_str)


        ROC = get_ROC(x_train, y_train, x_test, y_test, model,name, classifier)
        ROC_table.append( ROC)
        #dump_file_name = model_file_name + '_' + name + hd.Model_File_Extension
        #with open(dump_file_name, 'wb') as f_export:
        #    pickle.dump(classifier, f_export, pickle.HIGHEST_PROTOCOL)

    plot_ROC(ROC_table, detection)

    warnings.filterwarnings("always")
  
def PCA_Classification(real_data_file, gene_data_file =None, is_gene_data=True):

    title = 'PCA'
    
    with open(real_data_file, "rb") as f:
        freqs, x_real, y_real = pickle.load(f)
        if abs(np.mean(x_real)) > 1:
            x_real = (1-(x_real/-40))

    n_components = 2

    if is_gene_data == True:
        with open(gene_data_file, "rb") as f:
            freqs, x_gene, y_gene = pickle.load(f)
            if abs(np.mean(x_gene)) > 1:
                x_gene = (1-(x_gene/-40))

        x_real = x_real[np.where(y_real==hd.Person_Present_No)[0]]
        y_real = np.zeros(len(x_real), dtype=np.int)
        x = np.concatenate((x_real,x_gene), axis=0)
        y = np.concatenate((y_real, y_gene), axis=0)
    else:
        x = x_real
        y = y_real

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(x)

    colors = ['red', 'royalblue']
    target_names = ['Occupied', 'Unoccupied']

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    color=color, lw=2, label=target_name)

    #plt.title( title)
    plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize = legend_fontsize)
    if is_gene_data==True:
        fig_name = os.path.join(gene_data_dir, "gene_PCA_" + test_location + hd.JPG_File_Extension)
    else:
        fig_name = os.path.join(real_data_dir, "real_PCA_" + test_location + hd.JPG_File_Extension)
    plt.savefig(fig_name, dpi =400)
    #plt.show()
    plt.close()

def Train_CNN_Gene():
    with open(real_data_file_test, "rb") as f:
        freqs, x, y = pickle.load(f)
        if abs(np.mean(x))> 1:    
            x=(1-(x/-40))

    with open(gene_data_file, "rb") as f:
        freqs, x_test, y_test = pickle.load(f)
        if abs(np.mean(x_test))> 1:    
            x_test=(1-(x_test/-40))

    x_real_No = x[np.where(y==hd.Person_Present_No)[0]]
    y_real_No = np.zeros(len(x_real_No), dtype=np.int)
    x_real_Yes = x[np.where(y==hd.Person_Present_Yes)[0]]
    y_real_Yes = np.ones(len(x_real_Yes), dtype=np.int)

    x_real_No_num_train = int(len(x_real_No)*0.7)
    x_real_Yes_num_train = int(len(x_real_Yes)*0.7)

    x_train = np.concatenate((x_real_No[:x_real_No_num_train],x_real_Yes[:x_real_Yes_num_train]), axis=0)
    y_train = np.concatenate((np.zeros(x_real_No_num_train, dtype=np.int),np.ones(x_real_Yes_num_train, dtype=np.int)), axis=0 )

            
    x_test = np.concatenate((x_test, x_real_No[x_real_No_num_train:]), axis=0)
    y_test = np.concatenate((y_test, y_real_No[x_real_No_num_train:]), axis=0)

    s = np.arange(len(x_test))
    np.random.shuffle(s)
    x_test = x_test[s]
    y_test = y_test[s]

    s = np.arange(len(x_train))
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]

    print_str = 'Train data: {} real occupied: {} real unoccupied: {}\nTest data: {} gene occupied: {} real unoccupied: {}\n'
    print(print_str.format(len(x_train),
                    len(np.where(y_train==hd.Person_Present_Yes)[0]),
                    len(np.where(y_train==hd.Person_Present_No)[0]),
                    len(x_test),
                    len(np.where(y_test==hd.Person_Present_Yes)[0]),
                    len(np.where(y_test==hd.Person_Present_No)[0])
                    ))

    y_vec_train = np.zeros((len(y_train), class_num), dtype=np.float)
    for i, lable in enumerate(y_train):
      y_vec_train[i,y_train[i]] = 1.0

    y_vec_test = np.zeros((len(y_test), class_num), dtype=np.float)
    for i, lable in enumerate(y_test):
      y_vec_test[i,y_test[i]] = 1.0

    Detector_CNN.Train(70, 4, CNN_Folder, len(freqs), x_train, y_train, x_test, y_test )

def Test_CNN_Gene():
    with open(real_data_file_test, "rb") as f:
        freqs, x, y = pickle.load(f)
        if abs(np.mean(x))> 1:    
            x=(1-(x/-40))

    with open(gene_data_file, "rb") as f:
        freqs, x_test, y_test = pickle.load(f)
        if abs(np.mean(x_test))> 1:    
            x_test=(1-(x_test/-40))

    x_real_No = x[np.where(y==hd.Person_Present_No)[0]]
    y_real_No = np.zeros(len(x_real_No), dtype=np.int)
    x_real_Yes = x[np.where(y==hd.Person_Present_Yes)[0]]
    y_real_Yes = np.ones(len(x_real_Yes), dtype=np.int)

    x_real_No_num_train = int(len(x_real_No)*0.7)
    x_real_Yes_num_train = int(len(x_real_Yes)*0.7)

    x_train = np.concatenate((x_real_No[:x_real_No_num_train],x_real_Yes[:x_real_Yes_num_train]), axis=0)
    y_train = np.concatenate((np.zeros(x_real_No_num_train, dtype=np.int),np.ones(x_real_Yes_num_train, dtype=np.int)), axis=0 )

            
    x_test = np.concatenate((x_test, x_real_No[x_real_No_num_train:]), axis=0)
    y_test = np.concatenate((y_test, y_real_No[x_real_No_num_train:]), axis=0)

    s = np.arange(len(x_test))
    np.random.shuffle(s)
    x_test = x_test[s]
    y_test = y_test[s]

    s = np.arange(len(x_train))
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]


    y_vec_train = np.zeros((len(y_train), class_num), dtype=np.float)
    for i, lable in enumerate(y_train):
      y_vec_train[i,y_train[i]] = 1

    y_vec_test = np.zeros((len(y_test), class_num), dtype=np.float)
    for i, lable in enumerate(y_test):
      y_vec_test[i,y_test[i]] = 1

    TP, TN, FP, FN, accuracy = Detector_CNN.Test( x_test, y_test, CNN_Folder,len(freqs), )
    accuracy = "{:.2f}".format(accuracy)
    com.Log_Setup(CNN_Folder + '/log/', "test")
    com.Log_Info('----------------------------------------------------')
    print_str = 'CNN Model is trained with: {} Testing on: {}'.format(train_location, test_location)
    print(print_str)
    com.Log_Info(print_str)

    print_str = 'Train data: {} real occupied: {} real unoccupied: {}\nTest data: {} gene occupied: {} real unoccupied: {}\n'.format(len(x_train),
                    len(np.where(y_train==hd.Person_Present_Yes)[0]),
                    len(np.where(y_train==hd.Person_Present_No)[0]),
                    len(x_test),
                    len(np.where(y_test==hd.Person_Present_Yes)[0]),
                    len(np.where(y_test==hd.Person_Present_No)[0])
                    )
    print(print_str)
    com.Log_Info(print_str)
    print_str = "{:<12}{}".format ("Model Name", ":    TP     TN     FP    FN    Precision    Recall    F1        Accuracy")
    print(print_str)
    com.Log_Info(print_str)

    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    recall = TP / (TP + FN)

    if precision + recall>0:
        f1_value = 2 * precision * recall / (precision + recall)
    else:
        f1_value = 0
    precision = "{:.2f}".format(precision*100.0)
    recall = "{:.2f}".format(recall*100.0)
    f1_value = "{:.2f}".format(f1_value*100.0)

    print_str = '{:<12}:  {:>5}  {:>5} {:>5} {:>5}     {:>6}     {:>6}     {:>6}     {:>6}'.format("CNN", TP, TN, FP, FN, precision, recall, f1_value, accuracy)
    print(print_str)
    com.Log_Info(print_str)

def Train_CNN_Real():
    with open(real_data_file_test, "rb") as f:
        freqs, x, y = pickle.load(f)
        if abs(np.mean(x))> 1:    
            x=(1-(x/-40))

    with open(gene_data_file, "rb") as f:
        freqs, x_test, y_test = pickle.load(f)
        if abs(np.mean(x_test))> 1:    
            x_test=(1-(x_test/-40))

    trian_sample_num = int(len(x)/2 * 0.7)
    test_sample_num = len(x) - trian_sample_num

    power_Yes_train = x[np.where(y==1)[0]][:trian_sample_num]
    power_No_train = x[np.where(y==0)[0]][:trian_sample_num]

    label_Yes_train = np.ones(trian_sample_num, dtype=np.int)
    label_No_train = np.zeros(trian_sample_num, dtype=np.int)

    x_train = np.concatenate((power_Yes_train, power_No_train), axis=0)
    y_train = np.concatenate( (label_Yes_train, label_No_train), axis=0)
    s = np.arange(len(x_train))
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]

    power_Yes_test = x[np.where(y==1)[0]][trian_sample_num:]
    power_No_test = x[np.where(y==0)[0]][trian_sample_num:]

    label_Yes_test = np.ones(len(power_Yes_test), dtype=np.int)
    label_No_test = np.zeros(len(power_No_test), dtype=np.int)

    x_test = np.concatenate((power_Yes_test, power_No_test), axis=0)
    y_test = np.concatenate( (label_Yes_test, label_No_test), axis=0)
    s = np.arange(len(x_test))
    np.random.shuffle(s)
    x_test = x_test[s]
    y_test = y_test[s]

    print_str = 'Train data: {} real occupied: {} real unoccupied: {}\nTest data: {} gene occupied: {} real unoccupied: {}\n'
    print(print_str.format(len(x_train),
                    len(np.where(y_train==hd.Person_Present_Yes)[0]),
                    len(np.where(y_train==hd.Person_Present_No)[0]),
                    len(x_test),
                    len(np.where(y_test==hd.Person_Present_Yes)[0]),
                    len(np.where(y_test==hd.Person_Present_No)[0])
                    ))

    y_vec_train = np.zeros((len(y_train), class_num), dtype=np.float)
    for i, lable in enumerate(y_train):
      y_vec_train[i,y_train[i]] = 1

    y_vec_test = np.zeros((len(y_test), class_num), dtype=np.float)
    for i, lable in enumerate(y_test):
      y_vec_test[i,y_test[i]] = 1

    Detector_CNN.Train(60, 4, CNN_Folder, len(freqs), x_train, y_train, x_test, y_test )

if __name__ == '__main__':

    #Draw_One_Power(real_data_file_test, test_location)
    Draw_Average_Power_gene_real(gene_data_file,real_data_file_test )
    #Test_CNN_Gene()
    Train(real_data_file_test, test_data_file=gene_data_file, detection= Detection.Train_Real_Test_Gene)
    #Train(real_data_file_test, test_data_file=None, detection= Detection.Train_Real_Test_Real)
    #gene_loss_figure()
    '''
    Draw_Average_Power(gene_data_file, is_gene_data=True)
    Draw_Average_Power(real_data_file_test, is_gene_data=False)
   
    #Train_CNN_Gene()
    PCA_Classification(real_data_file_test, gene_data_file, is_gene_data=True)
    #Train(real_data_file_test, test_data_file=None, detection= Detection.Train_Real_Test_Real)
    
    #

    PCA_Classification(real_data_file_test, is_gene_data=False)
    Draw_Average_Power(real_data_file_test, is_gene_data=False)
    Train(real_data_file_test, test_data_file=None, detection= Detection.Train_Real_Test_Real)
    Train(gene_data_file, test_data_file=None, detection= Detection.Train_Gene_Test_Gene)
    '''