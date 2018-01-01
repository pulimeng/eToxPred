# -*- coding: utf-8 -*-
"""
Grid search for parameters

@author: Limeng Pu
"""

from __future__ import division

import scipy.io as sio
import numpy as np
from itertools import cycle

#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold

content = sio.loadmat('toxicity_over.mat')
y = content['y']
y = np.ravel(y)
X = content['X']

content = sio.loadmat('tox4test.mat')
yt = content['y']
yt = np.ravel(yt)
Xt = content['X']

md = 80
#md = md.astype('uint8')
ml = np.linspace(5,20,16)
ml = ml.astype('uint8')
ms = np.linspace(5,20,16)
ms = ms.astype('uint8')

size = [19,16,16]

tprs = np.zeros((16,16))
fprs = np.zeros((16,16))
mccs = np.zeros((16,16))

def predict_th(probs,th):
    pred = np.zeros([len(probs),len(th)])
    for i in range(len(probs)):
        for k in range(len(th)):
            temp_th = th[k]
            if probs[i][1] > temp_th:
                pred[i][k] = 1
            else:
                pred[i][k] = 0
    return pred
    
def compute_mcc(true,pred):
    mcc = np.zeros([pred.shape[1],])
    for k in range(pred.shape[1]):
        temp_pred = pred[:,k]
        temp_mcc = matthews_corrcoef(true,temp_pred)
        mcc[k] = temp_mcc
    return mcc
    
cv = StratifiedKFold(n_splits = 6)
#for i in range(len(md)):
for j in range(len(ml)):
    for k in range(len(ms)):
        temp_md = md
        temp_ml = ml[j]
        temp_ms = ms[k]
        ext = RandomForestClassifier(n_estimators = 100, max_depth = temp_md,
                           min_samples_leaf = temp_ml,
                           min_samples_split = temp_ms)
        mcc_tr = np.zeros([101,])
        ths = np.linspace(0,1,101)
        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        i = 0
        for (train, test), color in zip(cv.split(X, y), colors):
            probas_tr = ext.fit(X[train], y[train]).predict_proba(X[test])
            pred = predict_th(probas_tr,ths)
            pred = pred.astype('uint8')
            true = y[test]
            mcc_tr += compute_mcc(true,pred)
            i += 1    
        mean_mcc_tr = mcc_tr/6
        idx = list(mean_mcc_tr).index(max(mean_mcc_tr))
        print("Min samples leaf %i, Min samples split %i" % (temp_ml,temp_ms))
        probas_ext = ext.fit(X,y).predict_proba(Xt)
        pred = predict_th(probas_ext,[ths[idx]])
        pred = pred.astype('uint8')
        cnf = confusion_matrix(yt,pred)
        mcc_ext = matthews_corrcoef(yt,pred)
        TP1 = cnf[1][1]
        FP1 = cnf[0][1]
        FN1 = cnf[1][0]
        TN1 = cnf[0][0]
        fpr1 = float(FP1/(FP1+TN1))
        tpr1 = float(TP1/(TP1+FN1))
        mccs[j,k] = mcc_ext
        tprs[j,k] = tpr1
        fprs[j,k] = fpr1
        
sio.savemat('rf60results.mat', {'mccs': mccs,'tprs': tprs,'fprs': fprs})