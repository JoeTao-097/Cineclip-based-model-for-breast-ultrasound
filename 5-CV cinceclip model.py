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
os.environ['CUDA_VISIBLE_DEVICES']='0'
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



from tensorflow.keras import layers,Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

Earlystop_patience = 10
earlystopper = EarlyStopping(patience=Earlystop_patience, verbose=1)

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


def evaluate_mlp(PatientID, Patho, X,y,dfi,model_train=True, name=None,input_shape=None):
    PatientID = np.array(PatientID)
    X = np.array(X)
    y = np.array(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
    thresh = []
    for train, test in cv.split(PatientID, Patho):
        train = list(dfi[dfi['PatientID'].isin(PatientID[train])].index)
        test = list(dfi[dfi['PatientID'].isin(PatientID[test])].index)
        import tensorflow as tf
        model = MLP(input_shape)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      loss=['binary_crossentropy'],
                      metrics=['accuracy'])
        model_checkpoint = ModelCheckpoint('./MLP/{}_{}_temp.h5'.format(name,i), monitor='val_loss', verbose=1, mode='auto',
                                           save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss',
                                      factor=0.1,
                                      patience=5,
                                      verbose=1,
                                      mode='auto',
                                      min_delta=0.0001,
                                      cooldown=0,
                                      min_lr=0)
        X_train = X[train]
        X_test = X[test]
        Y_train = y[train]
        Y_test = y[test]
        if model_train == True:
            model.fit(x = X_train,y = Y_train, validation_data=(X_test,Y_test), epochs=60,batch_size=32,
                                     callbacks=[earlystopper, model_checkpoint, reduce_lr])
            probas_ = model.predict(X_test)
        df_test = dfi.iloc[test].copy()
        df_test.reset_index(drop=True, inplace=True)
        df_test['pred'] = probas_[:, 1]
        IDs = df_test['PatientID'].unique()
        y_true = []
        y_pred = []
        for ID in IDs:
            df_temp = df_test[df_test['PatientID'] == ID]
            df_temp.reset_index(drop=True, inplace=True)
            y_true.append(df_temp['Malignancy'][0])
            y_pred.append(np.max(df_temp['pred']))
        # collect TP TN FP FN
        thresh_0 = get_auc(0, np.array(y_true), np.array(y_pred), 'Malignancy', plot=False)
        thresh.append(thresh_0)
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
        valid_result.append(df_test)
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
    df = valid_result[0]
    for i in range(1, len(valid_result)):
        df = pd.concat([df, valid_result[i]])

    return df,thresh

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Dense
"""
fc_weight= np.load('./MLP_layer_weight/Dense_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
W = W1-W0

df = pd.read_excel('./Dataset.xlsx')
df.sort_values(by='PatientID', inplace=True)

df.reset_index(drop=True, inplace=True)
X = []
Y = []
map = [[1,0],[0,1]]
for idx in range(len(df)):
    print(idx)
    fmap = np.load(df.loc[idx,'Pfolder']+ 'vid_fmap_GAP_densenet.npy')
    X.append(W*np.max(fmap,0))
    Y.append(map[df.loc[idx,'Malignancy']])
'''
5-fold Eval
'''

df_ = df.drop_duplicates(subset=['PatientID'])
df_.reset_index(drop=True, inplace=True)
PatientID = list(df_['PatientID'])
Patho = list(df_['Malignancy'])
df_cv,thresh_dense = evaluate_mlp(PatientID, Patho, X, Y, dfi=df, name='dense', input_shape=(1024))
# df_cv.to_excel('./Result/Densenet_video_crossvalidation.xlsx', index=False, header=True)
#
"""
Mobile
"""

fc_weight= np.load('./MLP_layer_weight/Mobile_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
W = W1-W0

df = pd.read_excel('./Dataset.xlsx')
df_ = pd.read_excel('./Dataset_img.xlsx')

df = df[df['PatientID'].isin(df_['PatientID'])]
df = df[df['Usage']=='train']
df.reset_index(drop=True, inplace=True)

X = []
Y = []

map = [[1,0],[0,1]]
for idx in range(len(df)):
    print(idx)
    fmap = np.load(df.loc[idx,'Pfolder']+ 'vid_fmap_GAP_mobilenet.npy')
    X.append(W*np.max(fmap,0))
    Y.append(map[df.loc[idx,'Malignancy']])
'''
5-fold Eval
'''

df_ = df.drop_duplicates(subset=['PatientID'])
df_.reset_index(drop=True, inplace=True)
PatientID = list(df_['PatientID'])
Patho = list(df_['Malignancy'])
df_cv,thresh_mobile = evaluate_mlp(PatientID, Patho, X, Y, dfi=df,name='mobile',input_shape=(1024))
# df_cv.to_excel('./Result/MobileNet_video_crossvalidation.xlsx', index=False, header=True)
#
#
"""
Xception
"""

fc_weight= np.load('./MLP_layer_weight/xception_fc.npy',allow_pickle=True)
W1 = fc_weight[0][:,1]
W0 = fc_weight[0][:,0]
W = W1-W0

df = pd.read_excel('./Dataset.xlsx')
df_ = pd.read_excel('./Dataset_img.xlsx')

df = df[df['PatientID'].isin(df_['PatientID'])]
df = df[df['Usage']=='train']
df.reset_index(drop=True, inplace=True)
X = []
Y = []

map = [[1,0],[0,1]]
for idx in range(len(df)):
    print(idx)
    fmap = np.load(df.loc[idx,'Pfolder']+ 'vid_fmap_GAP_xception.npy')
    X.append(W*np.max(fmap,0))
    Y.append(map[df.loc[idx,'Malignancy']])
'''
5-fold Eval
'''

df_ = df.drop_duplicates(subset=['PatientID'])
df_.reset_index(drop=True, inplace=True)
PatientID = list(df_['PatientID'])
Patho = list(df_['Malignancy'])
df_cv,thresh_xce = evaluate_mlp(PatientID, Patho, X, Y, dfi=df,name='xce',input_shape=(2048))
# df_cv.to_excel('./Result/Xception_crossvalidation.xlsx', index=False, header=True)
# print(len(df_cv['PatientID'].unique().tolist()))