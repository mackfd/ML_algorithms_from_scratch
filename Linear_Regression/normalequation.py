#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np

class LinearRegressionNF:
    def fit(self, train, target):
        # to initial matrix of features i am adding one column of 1s that related to w0 or so called bias
        # to get correct number of 1s use the shape of training dataset
        # particularly the number of rows -> train.shape[0]
        # to add a column apply np.ones from numpy. BUT! 
        # use two brakets to get a column of 1s
        
        train_expanded = np.concatenate((np.ones((train.shape[0],1)),train.to_numpy()),axis=1)
        
        # this is "Normal equation formula" 
        # applying this will give us the best weights (vector of w) for predictions 
        
        self.w_best = np.linalg.inv(train_expanded.T.dot(train_expanded)).dot(train_expanded.T).dot(target)
        
        print('w best is: ', self.w_best)
        
    def predict(self, features):
        # here we are doing prediction 
        # first we need to transform input matrix
        # we have to add w0 column consisting of 1s like in a model training steps
        
        features_expanded = np.concatenate((np.ones((features.shape[0],1)),features.to_numpy()),axis=1)
        
        # to predict we simply multiply  modified (expanded by w0 = 1) input matrix by best weights w_best
        y_hat = features_expanded.dot(self.w_best)
        
        return y_hat

