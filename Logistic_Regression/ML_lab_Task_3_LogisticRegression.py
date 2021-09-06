#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[8]:


class Logistic_regression_grad():
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y, h, epsilon: float = 1e-5):
         
        # calculate binary cross entropy as loss 
        loss = (1/self.batch_size)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
        
        return loss
    
    def train(self, x, y, epochs, batch_size, lr):
        row, col = x.shape
        
        # Initializing weights and bias
        self.w = np.zeros((col,1))
        self.w0 = 1
        self.batch_size = batch_size
        self.lr = lr
        
        # defining bath size
        num_batches = x.shape[0]//self.batch_size
        
        
        for epoch in range(epochs):
            print("epoch: ", epoch)
            for batch_num in range(num_batches+1):
                
                # slicing batch data
                start = batch_num * self.batch_size
                end = (batch_num + 1) * self.batch_size
                
                x_batched = x[start:end]
                y_batched = np.array(y[start:end]).reshape((-1, 1))
                
                # predict for the epoch and batch
                # at first iteration we are using initial w/theta
                y_hat = self._sigmoid(np.dot(x_batched, self.w) + self.w0)
                
                # calculate gradient for weigths/theta
                error = y_hat - y_batched
        
                gradient_w = (1/self.batch_size)*np.dot(x_batched.T, error)
                gradient_w0 = (1/self.batch_size)*np.sum(error) 
                
                # adjusting weights/theta with learning rate annd calculated gradient 
                self.w -= self.lr*gradient_w
                self.w0 -= self.lr*gradient_w0
                
            # loss compute per epoch
            loss = self._compute_loss(y, self._sigmoid(np.dot(x, self.w) + self.w0))
            print("loss: ",loss)
        
        return self.w, self.w0        
    
    def predict(self, x_test):
        
        # predict on text data with calculated weigths/theta
        predictions = self._sigmoid(np.dot(x_test, self.w) + self.w0)
        
        # rounding up values to get classes
        predictions = np.round(predictions)
        
        return predictions

