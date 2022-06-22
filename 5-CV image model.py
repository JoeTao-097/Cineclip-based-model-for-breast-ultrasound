#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:35:54 2021

@author: joe
"""
import sklearn.svm

from base_tool import *
from eval_tool import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
import joblib
import  tensorflow as tf
def focal_loss(y_true, y_pred):
   gamma = 2.0
   alpha = 0.5
   pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
   pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
   return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

'''
feature reduce method
'''
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from scipy import interp
from sklearn.model_selection import KFold

def softmax(x):
    """ softmax function """
    # x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x))
    return x

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def evaluate(PatientID, Patho, X,y):
    PatientID = np.array(PatientID)
    X = np.array(X)
    Y = np.array(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    # cv = KFold(n_splits=3, shuffle=True, random_state=0)
    tprs = []
    aucs = []
    colors = ['lightcoral', 'peachpuff', 'darkkhaki', 'aquamarine', 'thistle']
    mean_fpr = np.linspace(0, 1, 100)
    valid_result = []
    i = 0
    # In order to split by PatientID, we don't split X,y directly
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(6, 4))
    for train, test in cv.split(PatientID, Patho):
        # test = list(dfi[dfi['PatientID'].isin(PatientID[test])].index)
        y_pred = X[test]
        y_true = Y[test]
        thresh_0 = get_auc(0, np.array(y_true), np.array(y_pred), 'Malignancy', plot=False)
        print(thresh_0)
        y_pred_comp_lvl = [1 if y > thresh_0 else 0 for y in y_pred]
        cm_comp = confusion_matrix(y_true, y_pred_comp_lvl)
        # TP.append()

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #        plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #                 label='ROC fold %d (AUC = %.2f)' % (i, roc_auc))
        i += 1
        plt.plot(fpr, tpr, color=colors[i - 1], label='Fold {} ROC (AUC = {:.3f} )'.format(i, roc_auc), lw=2,
                 alpha=.8)
    # sns.set_style('ticks')
    # fig, ax = plt.subplots(figsize=(6, 4))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.legend(loc="lower right")
    ax.grid(b=True, which='major', color='grey', linestyle='-')

    return df

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
DenseNet
"""
df0 = pd.read_excel('./Dataset.xlsx')
df = pd.read_excel('./Dataset_img.xlsx')
df = df[df['PatientID'].isin(df0['PatientID'])]
df = df[df['Usage']=='train']
df.sort_values(by='PatientID', inplace=True)
df.reset_index(drop=True, inplace=True)

fc_weight= np.load('./MLP_layer_weight/Dense_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
B1 = fc_weight[1][1]
B0 = fc_weight[1][0]

y_pred = []
y_true = []

for i in range(len(df)):
    fmap = np.load(df.loc[i,'Pfolder']+ 'imgs_fmap_GAP_densenet.npy')
    output = softmax([np.matmul(W0,fmap.T)+B0,np.matmul(W1,fmap.T)+B1])[1].item()
    y_pred.append(output)
df['pred'] = y_pred
IDs = df['PatientID'].unique()
y_true = []
y_pred = []
for ID in IDs:
    df_temp = df[df['PatientID'] == ID]
    df_temp.reset_index(drop=True, inplace=True)
    y_true.append(df_temp['Malignancy'][0])
    y_pred.append(np.mean(df_temp['pred']))

'''
5-fold Eval
'''
df_ = df.drop_duplicates(subset=['PatientID'])
df_.reset_index(drop=True, inplace=True)
PatientID = IDs
Patho = y_true
df_cv = evaluate(PatientID, Patho, y_pred, y_true)
# df_cv.to_excel('./Result/Densenet_PhysicianImages_crossvalidation.xlsx', index=False, header=True)
print(len(df_cv['PatientID'].unique().tolist()))

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

for i in range(len(df)):
    fmap = np.load(df.loc[i,'Pfolder']+ 'imgs_fmap_GAP_mobilenet.npy')
    output = softmax([np.matmul(W0,fmap.T)+B0,np.matmul(W1,fmap.T)+B1])[1].item()
    y_pred.append(output)
df['pred'] = y_pred
IDs = df['PatientID'].unique()
y_true = []
y_pred = []
for ID in IDs:
    df_temp = df[df['PatientID'] == ID]
    df_temp.reset_index(drop=True, inplace=True)
    y_true.append(df_temp['Malignancy'][0])
    y_pred.append(np.mean(df_temp['pred']))

'''
5-fold Eval
'''
df_ = df.drop_duplicates(subset=['PatientID'])
df_.reset_index(drop=True, inplace=True)
PatientID = IDs
Patho = y_true
df_cv = evaluate(PatientID, Patho, y_pred, y_true)
# df_cv.to_excel('./Result/MobileNet_PhysicianImages_crossvalidation.xlsx', index=False, header=True)
print(len(df_cv['PatientID'].unique().tolist()))

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

for i in range(len(df)):
    fmap = np.load(df.loc[i,'Pfolder']+ 'imgs_fmap_GAP_xception.npy')
    output = softmax([np.matmul(W0,fmap.T)+B0,np.matmul(W1,fmap.T)+B1])[1].item()
    y_pred.append(output)
df['pred'] = y_pred
IDs = df['PatientID'].unique()
y_true = []
y_pred = []
for ID in IDs:
    df_temp = df[df['PatientID'] == ID]
    df_temp.reset_index(drop=True, inplace=True)
    y_true.append(df_temp['Malignancy'][0])
    y_pred.append(np.mean(df_temp['pred']))

'''
5-fold Eval
'''
df_ = df.drop_duplicates(subset=['PatientID'])
df_.reset_index(drop=True, inplace=True)
PatientID = IDs
Patho = y_true
df_cv = evaluate(PatientID, Patho, y_pred, y_true)
# df_cv.to_excel('./Result/Xception_PhysicianImages_crossvalidation.xlsx', index=False, header=True)
print(len(df_cv['PatientID'].unique().tolist()))