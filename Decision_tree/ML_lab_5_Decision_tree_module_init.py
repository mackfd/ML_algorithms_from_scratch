#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from statistics import mode
class decesion_tree():
    
    def __init__(self,x ,y, max_depth):
        self.x = x
        self.y = y
        self.x_i = x.index
        self.y_i = y.index
        self.max_depth = max_depth
        self.to_split = []
        self.tree = {}
        self.depth = 0
        
    def build_tree(self):
    
        self.tree['root'] = {}
        self.to_split.append(self.x)
        self.tree['root']['root'] = self.y.values.flatten()
        #print(self.tree)
        #self.attributes = x.columns
        
        # stop building a tree?
        # if pureness(y) == 'Pure' and depth == 0:
        # return 'No tree to build'
        # for depth in range(self.max_depth+1):
        
        while len(self.to_split) > 0:
            for x in self.to_split:
                print('Depth:', self.depth)
                self.split(x)
        return self.tree
                
    def pureness(self, y):
        #if len(np.unique(y.values)) == 1:
        if len(np.unique(y)) == 1:
            return 'Pure'
        else:
            return 'Not pure'
    
    def entropy(self, p):

        size = p.shape[0]
        prob = p.value_counts()/size
       
        entropy = np.sum(-prob * np.log2(prob))
        pre_weight = entropy * size
        
        return size, pre_weight

    def split(self, x):
        #self.depth += 1
        attributes = x.columns
        y = self.y.iloc[list(x.index)]
        attr = self.find_attr(x, y, attributes)
        self.tree[attr] = {}
        print('attr found', attr)
        
        for branch in x[attr].unique():
            print(attr, branch)
     
            x_splitted = x[x[attr] == branch]
            
            x_splitted.drop([attr], inplace = True, axis=1)
            
            y_splitted = y.loc[y.index.isin(list(x[x[attr] == branch].index))].values.flatten()
            
            if self.pureness(y_splitted) == 'Pure':

                self.tree[attr][branch] = y_splitted[0]
                print('Pure')
            
            elif self.depth == self.max_depth:
                
                self.tree[attr][branch] = self.most_common_class(y_splitted)
                print('max depth')
                
            else:

                print('to split original', self.to_split)
                x_to_split = x_splitted.copy(deep = True)
                self.to_split.append(x_to_split)
                
        self.depth += 1
        self.to_split.pop(0)
        print('end of split call -> depth: ', self.depth)
        print('to split updated', self.to_split)
      

        return self.tree

    def find_attr(self, x, y, attributes):
        res  = float()
        x_size = x.shape[0]
        entropy_dict ={}

        for attr in attributes:
            res  = float()

            for col_value in x[attr].unique():

                p = y.loc[y.index.isin(list(x[x[attr] == col_value].index))]
                size, pre_weight = self.entropy(p)
                res += pre_weight/x_size
            entropy_dict[attr] = res
            
        return(min(entropy_dict, key = entropy_dict.get))
    
    def most_common_class(self, y):
    
        """
        :param y: the vector of class labels, i.e. the target
        returns: the most frequent class label in 'y'
        """
    
        return mode(y) 

