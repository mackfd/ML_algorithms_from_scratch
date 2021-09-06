#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
class svr_grd():
    '''
    Class of Support vector regression with epsilon insentive loss function.
    Optimization done with gradient descend. 
    Gradient calculated with batch processing. 
    !!! for big samples of data code must be adjusted into mini batch or SGD !!!
    For more details please use corresponding notebook
    
    Inmput of functions:
    
    __init__(epsilon, n_epochs, C=0.01, lr=0.01)
    fit(X, y)
    predict(X, w, w0)
    
    Output:
    
    train -> loss_res as a list , self.W, self.W0
    predict - > y_hat
    '''
    def __init__(self, epsilon, n_epochs, C=0.01, lr=0.01):
        self.epsilon = epsilon # the epsilon-insensitive loss, i.e. errors of less than  are ignored.
        self.C = C # the penalty term C controls the strengh of this penalty, and as a result, acts as an inverse regularization parameter 
        self.lr = lr 
        self.n_epochs = n_epochs
        
    def fit(self, X, y):
        
        # random initialization
        self.W = np.random.randn(X.shape[1], 1)/np.sqrt(X.shape[1]) # n feature weights
        #self._params = np.ones((num_features,num_targets))
        self.W0 = 0
        self.y = y
        self.loss_res = []
        m = int(len(y))
        
        for epoch in range(self.n_epochs):
            
            y_hat = X@self.W + self.W0
            loss = 1/2 * np.sum(self.W * self.W) + self.C * (np.sum(np.abs(y - y_hat)-self.epsilon)/ m)
            self.loss_res.append(loss)
            # derivative of loss function w.r.t to w and w0                                       
            gradient_W = self.W - self.C/m * np.sum((y - y_hat)/np.abs(y - y_hat) @ X)
            gradient_W0 = -self.C/m * np.sum((y - y_hat)/np.abs(y - y_hat))                                       
            self.W -= self.lr * gradient_W
            self.W0-= self.lr * gradient_W0  
                    
        return self.loss_res , self.W, self.W0
    def predict(self, X, w, w0):
        y_hat = X@w + w0
        return y_hat

