#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd

class simple_nn():
      '''
    This is simple nn class with 3 layers NN. In this class additional layer was added to the original layers
    from notebook given by Julian Stier and Sahib Julka.
    Moreover those functions were refactored so that final class would look more concise
    and easier to read. 
    Additionaly optimization were done to work with multiclassification tasks (i.e > than 2 classes)
    -----------------------------------------------------------------------------------------------
    OUTPUT:
    weights that must be used to call predict method of the class
    loss_res - list that consist of loss value calculated during training steps
    accuracy_res - list that consist of accuracy value calculated during training steps 
    -----------------------------------------------------------------------------------------------
    INPUT:
    creating a class examplar:
    
    simple_nn(input_dim, output_dim, lr, num_epochs, decay_rate)
    
    where: input_dim - input dimention of NN , 
                       output_dim - output dimention of NN, 
                       lr -learnin rate, 
                       num_epochs - number of epochs to iterate over 
                       decay_rate - decay rate for learning rate 
    For example:                    
    model = simple_nn(2, 2, 0.01, 2, 0.5)
    
    Once model is initialized, we can call train method  
    train(x, y, nn_hdim, batch_size)
    where: x, y are self-explanatory, 
           nn_hdim - num of neurons in hidden layer,
           batch_size - size of batch wich will be used to split the data in each epoch
    
    For example:     
    weights, loss_res, accuracy_res = model.train(X_train, y_train, 10, batch_size=50)
    ---------------------------------------------------------------------------------------
    PREDICT:
    Once model is trained it will return weights or also called "model".
    Having weights and x is sufficient to execute prediction with simple NN.
    Prediction will return predicted classes for the given inputs:
    
    y_hat = model.predict(weights, X_test)    
    '''
    
    def __init__(self, nn_input_dim, nn_output_dim, lr, epochs, decay_rate):
        
        self.nn_input_dim = nn_input_dim # input layer dimensionality
        self.nn_output_dim = nn_output_dim # output layer dimensionality
    
        self.lr_init = lr # learning rate for gradient descent
        self.epochs = epochs
        self.decay_rate = decay_rate # decay rate for calculating learninng rate decay         
        self.reg_lambda = 0.01 # regularization strength

    def init_weights(self, nn_hdim):
        np.random.seed(0)
        # when we initialize weights we normalise them by sqrt(n of input)
        # that has been empirically proved to improve the rate of convergence 
        
        self.W1 = np.random.rand(self.nn_input_dim, nn_hdim)/ np.sqrt(self.nn_input_dim)
        self.b1 = np.random.rand(1, nn_hdim)
        self.W2 = np.random.rand(nn_hdim, nn_hdim)/ np.sqrt(nn_hdim)
        self.b2 = np.random.rand(1, nn_hdim)
        
        # W3 and b3 are added as here we are having +1 layer 
        self.W3 = np.random.rand(nn_hdim, self.nn_output_dim)/ np.sqrt(nn_hdim)
        self.b3 = np.random.rand(1, self.nn_output_dim)    
        
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
    
    # sigmoid and sigmoid derivative have been added to this NN
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_deriv(self, x):
        f = 1/(1+np.exp(-x))
        df = f * (1 - f)
        return df
    
    def softmax(self, x):
        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    def tanh_deriv(self, x):
        return 1 - np.power(x, 2)
    
    def lr_decay(self, epoch):
        lr = self.lr_init/(1+self.decay_rate * epoch)
        return lr
    
    def forward_prop(self, W1, b1, W2, b2, W3, b3, x):
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        
        # layer 2 was added, i.e z2 and a2
        z2 = a1.dot(W2) + b2
        a2 = self.sigmoid(z2) 
        
        z3 = a2.dot(W3) + b3
        a3 = self.softmax(z3)

        return  z1, a1, z2, a2, z3, a3
    
    def backward_prop(self, z1, a1, z2, a2, z3, a3, W1, W2, W3, x, y):
        
        delta4 = a3 
        # so delta 4 is error that we want to dissiminate to W3, W2, W1
        # assigning to errors -1 ?
        delta4[range(self.batch_size), y] -= 1
        
        dW3 = (a2.T).dot(delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)
        
        # delta3 = error * by W3 * by sigmoid derivative
        delta3 = delta4.dot(W3.T) * self.sigmoid_deriv(a2)
        
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        # shouldn't we pass z1 to tanh_derivative? 
        delta2 = delta3.dot(W2.T) * self.tanh_deriv(a1)
        
        dW1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def params_update(self, W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3):
        
        dW3 += self.reg_lambda * W3
        dW2 += self.reg_lambda * W2
        dW1 += self.reg_lambda * W1
        
        W1 += -self.lr * dW1
        b1 += -self.lr * db1
        W2 += -self.lr * dW2
        b2 += -self.lr * db2
        W3 += -self.lr * dW3
        b3 += -self.lr * db3
                
        return W1, b1, W2, b2, W3, b3 
        
    def train(self, X, y, nn_hdim, batch_size):
    
    # Initialize the parameters to random values. We need to learn these.

        W1, b1, W2, b2, W3, b3  = self.init_weights(nn_hdim)   
        self.batch_size = batch_size
        loss_res = []
        accuracy_res = []
        
        # This is what we return at the end
        self.model = {}
        
        # defining number of batches 
        num_batches = X.shape[0]//self.batch_size
        
        # Gradient descent
        for epoch in range(0, self.epochs):
            
            print('epochs', epoch)
            if epoch == 0:
                self.lr = self.lr_init
            else:
                self.lr = self.lr_decay(epoch)
            
            for batch_num in range(num_batches):
                print('batch_num', batch_num)
          
                # slicing batch data
                start = batch_num * self.batch_size
                end = (batch_num + 1) * self.batch_size
                self.x_batched = X[start:end]
                self.y_batched = np.array(y[start:end])
                
                # training model by applying forward, backwar propagation and updating weithgs 
                z1, a1, z2, a2, z3, a3 = self.forward_prop(W1, b1, W2, b2, W3, b3, self.x_batched)
                dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(z1, a1, z2, a2, z3, a3, W1, W2, W3, self.x_batched, self.y_batched)
                W1, b1, W2, b2, W3, b3 = self.params_update(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3)
                              
                # Assign new parameters to the model
                self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
                
                # IMPORTANT
                # to compute loss value and accuracy we should use new weights and the same batch of x and y data 
                loss, acc = self.metrics(W1, W2, W3, b1, b2, b3, self.x_batched, self.y_batched)
                loss_res.append(loss)
                accuracy_res.append(acc)

        return self.model, loss_res, accuracy_res

    def metrics(self, W1, W2, W3, b1, b2, b3, X, y):
        
        z1, a1, z2, a2, z3, a3 = self.forward_prop(W1, b1, W2, b2, W3, b3, X)
        loss = self.calculate_loss(a3, y, W1, W2, W3)
        acc = self.calculate_accuracy(a3, y)
        return loss, acc
    
    def calculate_loss(self, a3, y, W1, W2, W3):

        corect_logprobs = -np.log(a3[range(self.batch_size), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))+np.sum(np.square(W3)))
        #print('loss a2',1./self.batch_size * data_loss)
        return 1./self.batch_size * data_loss   

    def calculate_accuracy(self, a3, y_true):

        y_hat = np.argmax(a3, axis=1)
        correct = sum(y_true == y_hat)
        incorrect = len(y_true) - correct
        return correct/len(y_true)*100
    
    def predict(self, model, x):
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
        # Forward propagation
        z1, a1, z2, a2, z3, a3 = self.forward_prop(W1, b1, W2, b2, W3, b3, x)
        return np.argmax(a3, axis=1)

