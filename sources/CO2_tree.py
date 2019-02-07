# coding: utf-8
'''
Created on 27 марта 2016 г.

@author: keen
'''
import decision_stamp as dst
from scipy.sparse import csr_matrix
from numpy import ndarray
from numpy import float64
from numpy import int64
from numpy import int8
from numpy import argmax
from numpy import zeros

from numpy import bincount
from numpy import nonzero
from numpy import transpose
from numpy import random
from numpy import asarray
from numpy import ones
from numpy import multiply
from numpy import unique
from numpy import count_nonzero

import sys

def expandMatrix(x):
        x = csr_matrix((x.data, x.indices, x.indptr),shape=(x.shape[0], x.shape[1] + 1), copy=False)
        tdat = [-1] * x.shape[0]
        tcols = [x.shape[1] - 1] *  x.shape[0]
        trows = range(x.shape[0])
        
        x_tmp = csr_matrix((tdat, (trows, tcols)),shape=x.shape,dtype = float64)
        
        x = x + x_tmp     
        
        return x


class CO2Tree:
    
    def clearNode(self,Y,sample_weight):
        y = asarray(multiply(sample_weight.reshape(-1),Y))
        
        y = y[nonzero(y)]
                
        differents = unique(y).shape[0]
        
        if differents <= 1:
            return True
        else:
            return False        
    
#protected  
    def buildTree(self, tol, C, x, Y, structure, sample_weight, features_weight, deth):
        if self.processed_counter > 0:
            if self.old_processed_counter != self.processed_counter:
                #print "Already processed: ", self.processed_counter
                self.old_processed_counter = self.processed_counter
        nnz = count_nonzero(sample_weight)
        if  self.max_deth is None or deth <= self.max_deth: 
            if nnz >= self.min_samples_split: 
                if not self.clearNode(Y, sample_weight):
                    #print "deth:", deth
                 
                    ds = dst.DecisionStamp(self.n_classes,self.class_max, features_weight,\
                                           self.kernel, self.sample_ratio,self.feature_ratio,\
                                           self.dual,C,tol,self.max_iter,self.gamma)
                    gres = ds.fit(x, Y, sample_weight,self.class_map,self.class_map_inv)
                    
                    if gres > 0.0:
                        self.nodes.append(ds)                        
                        id_ = len(self.nodes) - 1
    
                        tmp = [-1] * 2
                        structure.append(tmp) 
                        
                        #print "Tree size:", len(structure)
                        
                        sample_weightL = ds.sample_weightL   
                        sample_weightR = ds.sample_weightR
                        
                        features_weightL = ds.features_weight
                        features_weightR = ds.features_weight
                        
                        nnzL = count_nonzero(sample_weightL)
                        nnzR = count_nonzero(sample_weightR)
                        
                        #left
                        if nnzL > self.min_samples_leaf:     
                            structure[id_][0] = self.buildTree(tol, C, x, Y, structure, sample_weightL, features_weightL, deth + 1)
                        else:
                            self.processed_counter += nnzL
                        #right
                        if nnzR > self.min_samples_leaf:     
                            structure[id_][1] = self.buildTree(tol, C, x, Y, structure, sample_weightR, features_weightR, deth + 1)
                        else:
                            self.processed_counter += nnzR 
                        return id_
                    else:
                        self.processed_counter += nnz
                else:
                    self.processed_counter += nnz        
            else:
                self.processed_counter += nnz
        else:
            self.processed_counter += nnz                            
#        print "fin deth", deth            
        return -1                                 
        

#public   
    def fit(self,x,Y, sample_weight = None, preprocess = False):
        if isinstance(x,csr_matrix) and isinstance(Y,ndarray):
            if Y.shape[0] > 0 and x.shape[0] == Y.shape[0]:
                
                if preprocess:
                    x = expandMatrix(x)
                
                self.n_features = x.shape[1]
                classes_ = nonzero(bincount(Y))[0]
                self.n_classes = len(classes_)
                self.class_max = Y.max()
                
                self.class_map = zeros(shape = (self.class_max + 1), dtype = int64)
                self.class_map_inv = zeros(shape = (self.n_classes), dtype = int64)
                cc = 0
                for c in classes_:
                    self.class_map[c] = cc
                    self.class_map_inv[cc] = c                   
                    cc += 1 
                
                if sample_weight == None:
                    sample_weight = ones(shape=(1,x.shape[0]),dtype = int8)
                self.nodes = []

                dataf = [1] * x.shape[1]
                colsf = range(x.shape[1])
                rowsf = [0] * x.shape[1]
                features_weight = csr_matrix((dataf,(rowsf,colsf)) ,shape=(1,x.shape[1]),dtype = int8)
                
                self.structure = []
 
                self.total_len = self.buildTree(self.tol, self.C,x, Y, self.structure, sample_weight, features_weight, 0)
                
                #print "Tree ready: ", self.seed
                 
            else:
                print ("Wrong training set dimensionality")  
        else:
            print ("X type must be scipy.sparse.csr_matrix and Y type must be numpy.ndarray")          

    
    def predict(self,x, preprocess = False):
        probs = self.predict_proba(x, preprocess)
        
        return argmax(probs,axis=1)
   
    
    def predict_proba(self,x, preprocess = False):
        if isinstance(x,csr_matrix):
            res = zeros((x.shape[0], self.class_max + 1))
            
            if self.total_len > -1:
            
                if preprocess:
                    x = expandMatrix(x) 
                
                for i in range(x.shape[0]):
                    index = 0
                    old_index = 0
                    while index > -1:
                        old_index = index
                        if self.nodes[index].stamp_sign(x.getrow(i)) < 0:
                            index = self.structure[index][0]
                        else:
                            index = self.structure[index][1]
                        
                    res[i] =  self.nodes[old_index].predict_proba(x.getrow(i))
            return res             
        else:
            return "X type must be scipy.sparse.csr_matrix"   
                
    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_deth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = random.randint(1000), \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10.):
        self.max_deth = max_deth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.kernel = kernel
        self.tol = tol
        self.seed = seed
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.sample_ratio = sample_ratio

        self.feature_ratio=feature_ratio
        self.processed_counter = 0
        self.old_processed_counter = 0
        self.dual = dual
        
        random.seed(seed)
