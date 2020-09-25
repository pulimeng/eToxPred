# -*- coding: utf-8 -*-
"""
The last layer of the DBN serve as a linear regression instead of a logistic regression.
MSE is used as the cost function.
l-2 regularization is employed. The default setting of lamda1 and lamda2 are 0.01, which can be changed in line 52.

@author: Limeng Pu
"""

import theano
import numpy
import theano.tensor as T


class SemiLastLayer(object):
    def __init__(self, input, n_in, n_out):
        """
        :input:  input of the architecture (one mini-batch).
        :n_in: The number of input units, the dimension of the data space.
        :n_out: The number of output units, the dimension of the labels (here it's one).
        """

        # Initialize the weights to be all zeros.
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)

        # p_y_given_x forms a matrix, and y_pred will extract first element from each list.
        self.p_y_given_x = T.dot(input, self.W) + self.b

        # This caused a lot of confusion! It's basically the difference between [x] and x in python.
        self.y_pred = self.p_y_given_x[:,0]

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

    def squared_errors(self,y):
        """ Returns the mean of squared errors of the linear regression on this data. """
        #return  (T.mean(T.sqr(self.y_pred - y),axis=0))

        # return T.mean(T.sqr(self.y_pred - y),axis=1)
        return T.mean(T.sqr(self.y_pred - y))

    def reg_mse(self, y):
        """ Returns the mean of squared errors of the linear regression with l1 and l2 regularization on this data. """
        L1 = T.sum(abs(self.y_pred - y))
        L2_sqr = T.sum((self.y_pred - y)**2)
        return T.mean(T.sqr(self.y_pred - y)) + 0.01*L1 + 0.01*L2_sqr
