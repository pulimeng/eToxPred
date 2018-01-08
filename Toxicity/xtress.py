# -*- coding: utf-8 -*-
"""
Demo on trianing and testing the Extremely randomized trees using scikit-learn package

@author: Limeng Pu
"""

from __future__ import division

import pickle
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# load the .mat format data
with open('toxicity_train.pkl', 'rb') as train_file:
    train_data = pickle.load(train_file)
    X = train_data[0]
    y = train_data[1]
with open('toxicity_test.pkl', 'rb') as test_file:
    test_data = pickle.load(test_file)
    Xt = test_data[0]
    yt = test_data[1]

# random train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# configure the forest
ET = ExtraTreesClassifier(n_estimators = 30, max_features = 32,
                             max_depth = 5, min_samples_leaf = 1,
                             criterion = 'entropy',oob_score = True, bootstrap = True)

# fit the model to the training set
ET.fit(X_train, y_train)

# evaluations
tr_score = ET.score(X_train, y_train) # training set scores
te_score = ET.score(X_test, y_test) # test set scores
y_pred = ET.predict(X_test) # prediction results
cnf_matrix = confusion_matrix(y_test, y_pred) # confusion matrix
print("Training set score: %f" % tr_score)
print("Test set score: %f" % te_score)
x_score = np.mean(cross_val_score(ET, X, y, cv = 5)) # cross-validation scores
print("Average cross-validation score: %f" % np.mean(x_score))
y_prob = ET.predict_proba(X) # Tox-scores of compounds in X