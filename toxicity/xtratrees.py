# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:20:04 2017

@author: Limeng Pu
"""

# -*- coding: utf-8 -*-
"""
Random forest using scikit learn on balanced dataset 1015/1015 1024 bit

@author: Limeng Pu
"""

import scipy.io as sio
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

content = sio.loadmat('toxicity_over.mat')
y = content['y']
y = np.ravel(y)
X = content['X']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

randf = ExtraTreesClassifier(n_estimators = 30, max_features = 32,
                             max_depth = 5, min_samples_leaf = 1,
                             criterion = 'entropy',oob_score = True, bootstrap = True)

randf.fit(X_train, y_train)
tr_score = randf.score(X_train, y_train)
te_score = randf.score(X_test, y_test)
y_pred = randf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Training set score: %f" % tr_score)
print("Test set score: %f" % te_score)
x_score = np.mean(cross_val_score(randf, X, y, cv = 5))
print("Average cross-validation score: %f" % np.mean(x_score))
y_prob = randf.predict_proba(X)
#sio.savemat('y_prob_randforest.mat',{'y_prob': y_prob})