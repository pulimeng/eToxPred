# -*- coding: utf-8 -*-
"""
Predict the SAscore of a chemical compound with the trained DBN.
The trained model (.pkl file) is loaded at line 20

@author: Limeng Pu
"""

import pickle
import theano
import numpy
from theano.gof import graph
import theano.sandbox.cuda
import random
from sa_dbn import DBN

#device = 'cpu'
#theano.sandbox.cuda.use(device)

# load the data
with open('/home/limeng/Desktop/toxicity/SA score prediction/code/sa-part-resampled.pkl', 'rb') as in_strm:
    data = pickle.load(in_strm)

# select a compound for prediction
X = data[0][200]
X = numpy.reshape(X,(1,1024))
y = data[1][200]

# prediction function
def predict(X_test, filename='trained_model.pkl'):
    # load the saved model
    with open(filename, 'rb') as in_strm:
        regressor = pickle.load(in_strm)
    in_strm.close()
    y_pred = regressor.linearLayer.y_pred
    # find the input to theano graph
    inputs = graph.inputs([y_pred])
    # select only x
    inputs = [item for item in inputs if item.name == 'x']
    # compile a predictor function
    predict_model = theano.function(
        inputs=inputs,
        outputs=y_pred)
    X_test = X_test.astype(numpy.float32)
    predicted_values = predict_model(X_test)
    return predicted_values

# prediction
predicted_values = predict(X,'trained_model_cpu.pkl') # if cuda is not installed, use the trained_model_cpu
true_values = y
# the SAscore here is between 0 and 1 to suit the range of the activation function
# the following line converts the output to between 1 and 10
predicted_values = numpy.asscalar(predicted_values*10)
true_values = true_values*10
print('Predicted value: ' + str(predicted_values) +
        '\nTrue value: ' + str(true_values) +
        '\nAbsolute error: ' + str(numpy.asscalar(numpy.absolute(predicted_values-true_values))))
