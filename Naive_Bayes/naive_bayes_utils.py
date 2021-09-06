#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np

def metrics(y, predictions, metric_name):
    result = []
    y_hat = []
    if metric_name == 'Accuracy':
        for i in range(len(predictions)): 
            y_hat.append(max(predictions[i][list(predictions[i].keys())[0]], key=predictions[i][list(predictions[i].keys())[0]].get))
        for i in range(len(y)):
            result.append(int(y[i] == y_hat[i]))
        correct = sum(result)
        incorrect = len(y) - correct
        print("Correct: {}".format(correct))
        print("Incorrect: {}".format(incorrect))
        print("Accuracy: {:2.2%}".format(correct/len(y)))
    else:
        output = 'No such metric defined'
        print(output)

