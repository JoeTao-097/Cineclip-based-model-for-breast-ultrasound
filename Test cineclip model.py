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


def eval(X, Y, df, model,thresh=0.5):
    y_pred = []
    for x in X:
        x = np.expand_dims(x,0)
        y_pred.append(model.predict(x)[0][1])
    df['y_pred_allframe'] = y_pred
    UIDs = df['UID'].unique()
    y_true = []
    y_pred = []
    for UID in UIDs:
        df_temp = df[df['UID'] == UID]
        df_temp.reset_index(drop=True, inplace=True)
        y_true.append(df_temp['Malignancy'][0])
        y_pred.append(np.mean(df_temp['y_pred_allframe']))
    print(y_pred,y_true)
    thresh_0 = thresh
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

    return y_pred


from tensorflow.keras import layers,Model
def MLP(input_shape):
    inputs = layers.Input(input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(256)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(2)(x)
    # x = layers.Dropout(0.4)(x)
    x = layers.Activation(activation='softmax')(x)
    model = Model(inputs,x)
    return model


import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./Dataset.xlsx')
df_train = df[df['Usage']=='train']
df_test = df[df['Usage']=='Pressure Test']
df_test = df_test.sort_values(by=['UID'])
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

"""
Dense
"""
fc_weight= np.load('./MLP_layer_weight/Dense_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
W = W1-W0

X_train = []
Y_train = []
for idx in range(len(df_train)):
    print(idx)
    file = str(df_train.loc[idx,'vid'])
    save_dir = file.replace(file.split('/')[-1],'')
    fmap = np.load(save_dir + 'vid_fmap_GAP_densenet.npy')
    np.save(df_train.loc[idx,'Pfolder']+ 'vid_fmap_GAP_densenet.npy', fmap)
    fmap = np.load(df_train.loc[idx,'Pfolder']+ 'vid_fmap_GAP_densenet.npy')
    X_train.append(W*np.max(fmap,0))
    Y_train.append(df_train.loc[idx,'Malignancy'])

X_test = []
Y_test = []
for idx in range(len(df_test)):
    print(idx)
    file = str(df_test.loc[idx,'vid'])
    save_dir = file.replace(file.split('/')[-1],'')
    fmap = np.load(save_dir + 'vid_fmap_GAP_densenet.npy')
    np.save(df_test.loc[idx,'Pfolder']+ 'vid_fmap_GAP_densenet.npy', fmap)
    famp = np.load(df_test.loc[idx,'Pfolder']+ 'vid_fmap_GAP_densenet.npy')
    X_test.append(W*np.max(fmap,0))
    Y_test.append(df_test.loc[idx,'Malignancy'])

model = MLP((1024))
model.load_weights('./MLP/dense_2.h5')
P_dense = eval(X_test,Y_test,df_test,model, thresh=0.5)
#
#
"""
MobileNet
"""
fc_weight= np.load('./MLP_layer_weight/Mobile_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
W = W1-W0

X_train = []
Y_train = []
for idx in range(len(df_train)):
    print(idx)
    fmap = np.load(df_train.loc[idx,'Pfolder']+ 'vid_fmap_GAP_mobilenet.npy')
    X_train.append(W*np.max(fmap,0))
    Y_train.append(df_train.loc[idx,'Malignancy'])

X_test = []
Y_test = []
for idx in range(len(df_test)):
    print(idx)
    fmap = np.load(df_test.loc[idx,'Pfolder']+ 'vid_fmap_GAP_mobilenet.npy')
    X_test.append(W*np.max(fmap,0))
    Y_test.append(df_test.loc[idx,'Malignancy'])

model = MLP((1024))
model.load_weights('./MLP/mobile_4.h5')
P_mobile = eval(X_test,Y_test,df_test,model, thresh=0.5)

"""
Xception
"""
fc_weight= np.load('./MLP_layer_weight/xception_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
W = W1-W0

X_train = []
Y_train = []
for idx in range(len(df_train)):
    print(idx)
    fmap = np.load(df_train.loc[idx,'Pfolder']+ 'vid_fmap_GAP_xception.npy')
    X_train.append(W*np.max(fmap,0))
    Y_train.append(df_train.loc[idx,'Malignancy'])

X_test = []
Y_test = []
for idx in range(len(df_test)):
    print(idx)
    fmap = np.load(df_test.loc[idx,'Pfolder']+ 'vid_fmap_GAP_xception.npy')
    X_test.append(W*np.max(fmap,0))
    Y_test.append(df_test.loc[idx,'Malignancy'])

model = MLP((2048))
model.load_weights('./MLP/xce_2.h5')
P_xce = eval(X_test,Y_test,df_test,model, thresh=0.5)

df = df_test.drop_duplicates(subset=['UID'])
df.reset_index(drop=True,inplace=True)
df['DenseNet'] = P_dense
df['MobileNet'] = P_mobile
df['Xception'] = P_xce
# df.to_excel('./Dataset_RW_test.xlsx', index=False, header=True)
