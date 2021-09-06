#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pprint
import re
import codecs
import math
import itertools
import numpy as np

class NaiveBayesClassifier:
    '''
    NaiveBayes classifier class that works on newsgroups data 
    
    Input of functions:
     - train(path) : path consist of folders that considered categories. Each  folder has documents related to category
     - test(path_test): passing all documents to classify
     - predict(test_doc_path): pass one document to classify
    
    Otput:
    test - > predictions 
    (list of probabilities where index leads to a dictionary with key = document_id and values - probabilities)
    Example:
    predictions[77]
    >>> {'53417': {'alt.atheism': -1004.2773913093156,
               'comp.graphics': -1136.0621753579267,
               'comp.os.ms-windows.misc': -1226.033852615661,
               'comp.sys.ibm.pc.hardware': -1148.097576641767,
               'comp.sys.mac.hardware': -1145.137336934539,
               'comp.windows.x': -1133.879398672583,
               'misc.forsale': -1188.3876684286702,
               'rec.autos': -1128.8123602315757,
               'rec.motorcycles': -1133.5319831175282,
               'rec.sport.baseball': -1133.3682060575777,
               'rec.sport.hockey': -1154.4299641216624,
               'sci.crypt': -1099.6411178657333,
               'sci.electronics': -1144.8538186308606,
               'sci.med': -1105.897425977385,
               'sci.space': -1130.0437040323202,
               'soc.religion.christian': -1102.8861916273777,
               'talk.politics.guns': -1102.603818048012,
               'talk.politics.mideast': -1121.696588249512,
               'talk.politics.misc': -1108.649334060547,
               'talk.religion.misc': -1101.5161087894655}}
               
    to get the best score for a particula document
    max(predictions[1]['53257'], key=predictions[1]['53257'].get)
    >>> 'alt.atheism'
    
    predict - > scores
    (dictionary where key is document_id and values are probabilities to categories)
    
    '''
    def __init__(self):
        self.min_count = 1
        self.vocabulary = {}
        self.num_docs = 0
        self.classes = {}
        self.priors = {}
        self.conditionals = {}
        self.num_docs = 0
        #self.scores = {}
        
    def _tokenize_str(self, doc):
        return re.findall(r'\b\w\w+\b', doc) # return all words with #characters > 1
    
    def _tokenize_file(self, doc_file):
    # reading document and encoding
        with codecs.open(doc_file, encoding='latin1') as doc:
            # transforming all words into a lower register -> excluding register sensitivity 
            doc = doc.read().lower()
            # splitting with \n\n
            _header, _blankline, body = doc.partition('\n\n')
            # received body we are passing to tokenize_str
            return self._tokenize_str(body) # return all words with #characters > 1
        
    def train(self, path):
        
        #path_newsgroups_train -> path
        for class_name in os.listdir(path):
            # create a nested dictionary structure
            self.classes[class_name] = {"doc_counts": 0, "term_counts": 0, "terms": {}}
            #print(class_name)
            # getting path to folders of each class 
            path_class = os.path.join(path, class_name)
            # iteration of each document of each class via nested loop 
            for doc_name in os.listdir(path_class):
                # calling fuction tokenize and passing a path of each folder (i.e document)
                # as a result we will have a list of words from body of document
                terms = self._tokenize_file(os.path.join(path_class, doc_name))
                self.num_docs += 1
                self.classes[class_name]["doc_counts"] += 1
                # build vocabulary and count terms
                # term is one word from list of words from one document 
                for term in terms:
                    #print(term)
                    self.classes[class_name]["term_counts"] += 1
                    # for each class we are saving and counting how many times we see a term 
                    # in vocubulary dict it is flat dictionary -> total times we have seen a term in all documents  
                    # in classes dict it is nested dictionary -> numbner of times we have seen a term for a class 
                    if not term in self.vocabulary:
                        self.vocabulary[term] = 1
                        self.classes[class_name]["terms"][term] = 1
                    else:
                        self.vocabulary[term] += 1
                        if not term in self.classes[class_name]["terms"]:
                            self.classes[class_name]["terms"][term] = 1
                        else:
                            self.classes[class_name]["terms"][term] += 1
                
         # learning function 
        for cn in self.classes:
        # calculate priors
        # P(C = 11) = 600/20000 --> 0.03
        # log(P(C = 11)) = log(600)-log(20000)
        # P(y) - prior probability of classes
            self.priors[cn] = math.log(self.classes[cn]['doc_counts']) - math.log(self.num_docs)
            # calculate conditionals
            # better say P(X|y) - likelihood
            self.conditionals[cn] = {}
            # cdict - retrieving all words for one class 
            cdict = self.classes[cn]['terms']
            # then take values counted for each words and summing up 
            # c_len - sum over all number of worlds of the class 
            c_len = sum(cdict.values())
        
            for term in self.vocabulary:
                t_ct = 1.
                t_ct += cdict[term] if term in cdict else 0.
                self.conditionals[cn][term] = math.log(t_ct) - math.log(c_len + len(self.vocabulary))  
    
    def test(self, path_test):
        predictions = []
        # predict function 
        print("Testing <%s>" % path_test)
        # iterating over all classes from dictionary classes to fild documents paths 
        for class_num, class_name in enumerate(self.classes):
            # for each class constructing path to documents: 1 doc - 1 path
            for doc in os.listdir(os.path.join(path_test, class_name)):
                doc_path = os.path.join(path_test, class_name, doc)
                # once we get a path, go to tokens 
                token_list = self._tokenize_file(doc_path)
                result = self._scores(doc, token_list)
                predictions.append(result)
        return predictions
                
    def _scores(self, doc, tokens):
        scores = {}
        scores[doc] = {}
        for class_num, class_name in enumerate(self.classes):
            scores[doc][class_name] = self.priors[class_name]
            for term in tokens:
                    if term in self.vocabulary:
                        # retrieving P(x = term |y = class)
                        # result is  P(y= class) + sum over all terms[P(x = term |y = class)]
                        # souldn't it be log() operation ???
                        
                        # as a result we have "probatility" of a class for a given document 
                        #self.scores[doc][cn] += self.conditionals[cn][term]
                        scores[doc][class_name] += self.conditionals[class_name][term]
        return scores
                
    def predict(self, test_doc_path):
        # retreiving doc num from given path
        doc = test_doc_path.split('\\')[2]
        # tokenizing doc 
        token_list_predict = self._tokenize_file(test_doc_path)
        # calling score calculation for given doc 
        return self._scores(doc, token_list_predict)

