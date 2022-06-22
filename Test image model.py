"""
Created on Fri Oct 29 13:35:54 2021

@author: joe
"""
import sklearn.svm

from base_tool import *
from eval_tool import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
import joblib
import  tensorflow as tf

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def softmax(x):
    """ softmax function """
    # x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x))
    return x

df0 = pd.read_excel('./Dataset.xlsx')
df = pd.read_excel('./Dataset_img.xlsx')
df = df[df['PatientID'].isin(df0['PatientID'])]
df_train = df[df['Usage']=='train']
df_test = df[df['Usage']=='RW Test']
df_test = df_test.sort_values(by=['PatientID'])
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

fc_weight= np.load('./MLP_layer_weight/Dense_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
B1 = fc_weight[1][1]
B0 = fc_weight[1][0]

y_pred = []
y_true = []

for i in range(len(df_test)):
    fmap = np.load(df_test.loc[i,'Pfolder']+ 'imgs_fmap_GAP_densenet.npy')
    output = softmax([np.matmul(W0,fmap.T)+B0,np.matmul(W1,fmap.T)+B1])[1].item()
    y_pred.append(output)
df_test['pred'] = y_pred
IDs = df_test['PatientID'].unique()
y_true = []
y_pred = []
for ID in IDs:
    df_temp = df_test[df_test['PatientID'] == ID]
    df_temp.reset_index(drop=True, inplace=True)
    y_true.append(df_temp['Malignancy'][0])
    y_pred.append(np.mean(df_temp['pred']))

thresh_0 = get_auc(0, np.array(y_true), np.array(y_pred), 'Malignancy', plot=False)
thresh_0 = 0.4821
y_pred_comp_lvl = [1 if y > thresh_0 else 0 for y in y_pred]
cm_comp = confusion_matrix(y_true, y_pred_comp_lvl)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout(pad=2, w_pad=2.)
fig.set_figheight(8)
fig.set_figwidth(7)
thresh_0 = get_auc(axes[0, 0], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1')
thresh_AP = get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1')
plot_confusion_matrix(axes[1, 0], cm_comp, ["0", "1"], title='Malignancy', normalize=False)
plot_confusion_matrix(axes[1, 1], cm_comp, ["0", "1"], title='Malignancy (normalized)')
fig.show()
P_dense = y_pred


"""
MobileNet
"""
fc_weight= np.load('./MLP_layer_weight/Mobile_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
B1 = fc_weight[1][1]
B0 = fc_weight[1][0]


y_pred = []
y_true = []

for i in range(len(df_test)):
    fmap = np.load(df_test.loc[i,'Pfolder']+ 'imgs_fmap_GAP_mobilenet.npy')
    output = softmax([np.matmul(W0,fmap.T)+B0,np.matmul(W1,fmap.T)+B1])[1].item()
    y_pred.append(output)
df_test['pred'] = y_pred
IDs = df_test['PatientID'].unique()
y_true = []
y_pred = []
for ID in IDs:
    df_temp = df_test[df_test['PatientID'] == ID]
    df_temp.reset_index(drop=True, inplace=True)
    y_true.append(df_temp['Malignancy'][0])
    y_pred.append(np.mean(df_temp['pred']))

thresh_0 = get_auc(0, np.array(y_true), np.array(y_pred), 'Malignancy', plot=False)
thresh_0 = 0.4743
y_pred_comp_lvl = [1 if y > thresh_0 else 0 for y in y_pred]
cm_comp = confusion_matrix(y_true, y_pred_comp_lvl)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout(pad=2, w_pad=2.)
fig.set_figheight(8)
fig.set_figwidth(7)
thresh_0 = get_auc(axes[0, 0], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1')
thresh_AP = get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1')
plot_confusion_matrix(axes[1, 0], cm_comp, ["0", "1"], title='Malignancy', normalize=False)
plot_confusion_matrix(axes[1, 1], cm_comp, ["0", "1"], title='Malignancy (normalized)')
fig.show()
P_mobile = y_pred

"""
Xception
"""
fc_weight= np.load('./MLP_layer_weight/xception_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
B1 = fc_weight[1][1]
B0 = fc_weight[1][0]

y_pred = []
y_true = []

for i in range(len(df_test)):
    fmap = np.load(df_test.loc[i,'Pfolder']+ 'imgs_fmap_GAP_xception.npy')
    output = softmax([np.matmul(W0,fmap.T)+B0,np.matmul(W1,fmap.T)+B1])[1].item()
    y_pred.append(output)
df_test['pred'] = y_pred
IDs = df_test['PatientID'].unique()
y_true = []
y_pred = []
for ID in IDs:
    df_temp = df_test[df_test['PatientID'] == ID]
    df_temp.reset_index(drop=True, inplace=True)
    y_true.append(df_temp['Malignancy'][0])
    y_pred.append(np.mean(df_temp['pred']))

thresh_0 = get_auc(0, np.array(y_true), np.array(y_pred), 'Malignancy', plot=False)
thresh_0 = 0.3684
y_pred_comp_lvl = [1 if y > thresh_0 else 0 for y in y_pred]
cm_comp = confusion_matrix(y_true, y_pred_comp_lvl)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout(pad=2, w_pad=2.)
fig.set_figheight(8)
fig.set_figwidth(7)
thresh_0 = get_auc(axes[0, 0], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1')
thresh_AP = get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1')
plot_confusion_matrix(axes[1, 0], cm_comp, ["0", "1"], title='Malignancy', normalize=False)
plot_confusion_matrix(axes[1, 1], cm_comp, ["0", "1"], title='Malignancy (normalized)')
fig.show()
P_xce = y_pred
df_test.drop_duplicates(subset=['PatientID'], inplace=True)
df_test['DenseNet'] = P_dense
df_test['MobileNet'] = P_mobile
df_test['Xception'] = P_xce
# df_test.to_excel('./img_rw_test_voting2.xlsx', index=False, header=True)