#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[8]:


class LinearRegression_grad:
    
    '''
    lr - learning rate
    epochs - number of iterations to go through all observations one time = epoch 
    batch_size - recommended 100-200 observations 
    '''
    
    def __init__(self, lr, epochs, batch_size):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
    
    def train(self, x, y):

        # adding bias to initial fratures set as a column 
        x_expanded = np.concatenate((np.ones((x.shape[0],1)),x.to_numpy()),axis=1)
        
        # initialize weights for the given features set
        w = np.ones((x_expanded.shape[1],1))
        #w = np.zeros(x_expanded.shape[1])
        
        # defining number of steps within one epoch 
        num_batches = x_expanded.shape[0]//self.batch_size
        print('num_batches', num_batches)
        
        # nested loop, epochs plus number of batches as inner loop
        # calculatting loss and adjusting weights 
        
        for epoch in range(self.epochs):
            print('epoch:', epoch)
            for batch_num in range(num_batches):
                print('batch_num:',batch_num)
                
                # defining batch indexes to get piece of data
                start = batch_num * self.batch_size
                end = (batch_num + 1) * self.batch_size
                #print(start)
                #print(end)
                
                # slicing data equal to batch size
                x_expanded_batched = x_expanded[start:end]
                y_batched = np.array(y[start:end]).reshape((-1, 1))
                
                # predict for the given batch
                y_hat = x_expanded_batched.dot(w)
                
                error =  y_batched - y_hat
                
                mse = np.mean(error**2.)
                print('mse:', mse)
                
                #gradient = (-2 * (x_expanded_batched.T.dot(error)))/self.batch_size
                gradient = - (1./self.batch_size) * 2. * np.dot(x_expanded_batched.T, error)
                
                w -= gradient * self.lr
              
        self.w = w[1:]
        self.w0 = w[0]
        
        print('w:', self.w)
        print('w0:', self.w0)
        
        
    def predict(self, x_test):
        
        y_hat = x_test.dot(self.w) + self.w0
        
        return y_hat


# In[ ]:




