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
from numpy import dot
from numpy import random
from numpy import asarray
from numpy import ones
from numpy import multiply
from numpy import unique
from numpy import count_nonzero

from copy import deepcopy

import numpy

import sys

from memory_profiler import profile

def expandMatrix(x):
        x = csr_matrix((x.data, x.indices, x.indptr),shape=(x.shape[0], x.shape[1] + 1), dtype = numpy.float32, copy=False)
        tdat = [-1] * x.shape[0]
        tcols = [x.shape[1] - 1] *  x.shape[0]
        trows = range(x.shape[0])
        
        x_tmp = csr_matrix((tdat, (trows, tcols)),shape=x.shape,dtype = numpy.float32)
        
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

    #@profile
    def buildTree(self, tol, C, x, Y, structure, sample_weight, features_weight, deth,balanced,sam_counts):
        
        if self.processed_counter > 0:
            if self.old_processed_counter != self.processed_counter:
                #print "Already processed: ", self.processed_counter
                self.old_processed_counter = self.processed_counter
        nnz = count_nonzero(sample_weight)

        if  self.max_deth is None or deth <= self.max_deth: 
            if nnz >= self.min_samples_split: 
                if not self.clearNode(Y, sample_weight):
                    #print "deth:", deth
                    #balanced = True
                    cf = 1
                    #if deth > 6:
                    #    cf = 10

                    ds = dst.DecisionStamp(self.n_classes,self.class_max, features_weight,\
                                           self.kernel, self.sample_ratio,self.feature_ratio,\
                                           self.dual,C/cf,tol,self.max_iter,self.gamma,self.intercept_scaling,self.dropout_low,self.dropout_high,balanced,self.noise,self.cov_dr, self.criteria)

                    gres,sample_weightL, sample_weightR = ds.fit(x, Y, sample_weight,self.class_map,self.class_map_inv,sam_counts)
                    
                    if gres > 0.0:
                        
                        self.nodes.append(ds)                        
                        id_ = len(self.nodes) - 1
    
                        tmp = [-1] * 2
                        structure.append(tmp) 
                        
                        #print ("Tree size:", len(structure))

                        features_weightL = ds.features_weight
                        features_weightR = ds.features_weight
                        
                        nnzL = count_nonzero(sample_weightL)
                        nnzR = count_nonzero(sample_weightR)

                        #print ("Balance: ",float(nnzL)/(nnzL + nnzR),float(nnzR)/(nnzL+nnzR),deth) 
                        
                        #left
                        if nnzL > self.min_samples_leaf:    
                            if not (sam_counts is None): 
                                structure[id_][0] = self.buildTree(tol, C*self.intercept_scaling, x, Y, structure, sample_weightL, features_weightL, deth + 1,balanced,deepcopy(sam_counts))
                            else:
                                structure[id_][0] = self.buildTree(tol, C*self.intercept_scaling, x, Y, structure, sample_weightL, features_weightL, deth + 1,balanced,sam_counts)    
                        else:
                            self.processed_counter += nnzL
                        #right
                        if nnzR > self.min_samples_leaf:  
                            if not (sam_counts is None):    
                                structure[id_][1] = self.buildTree(tol, C*self.intercept_scaling, x, Y, structure, sample_weightR, features_weightR, deth + 1,balanced,deepcopy(sam_counts))
                            else:
                                structure[id_][1] = self.buildTree(tol, C*self.intercept_scaling, x, Y, structure, sample_weightR, features_weightR, deth + 1,balanced,sam_counts)                                    
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
                
                if self.cov_dr > 0:
                    sam_counts = zeros((x.shape[0]),dtype=int8)
                else:
                    sam_counts = None
                        
                self.total_len = self.buildTree(self.tol, self.C,x, Y, self.structure, sample_weight, features_weight, 1,True,sam_counts)
                

                for i,s in enumerate(self.structure):
                    if s[0] == -1 or s[1] == -1:
                        self.leaves.append(i) 
                
                #print "Tree ready: ", self.seed
                 
            else:
                print ("Wrong training set dimensionality")  
        else:
            print ("X type must be scipy.sparse.csr_matrix and Y type must be numpy.ndarray")          

    
    def predict(self,x, preprocess = False):
        probs = self.predict_proba(x, preprocess)
        
        return argmax(probs,axis=1)
   
    
    def predict_proba(self,x, Y = None,preprocess = False, stat_only = False, use_weight = True):
        if isinstance(x,csr_matrix):
            res = zeros((x.shape[0], self.class_max + 1))
            
            if self.total_len > -1:
            
                if preprocess:
                    x = expandMatrix(x) 
                
                old_indexes = {0:ones((x.shape[0],), dtype=bool)}

                final_estimators = {}

                while True:
                    new_indexes = {}
                    for index in old_indexes:
                        if index > -1:
                            x_shr = csr_matrix(x[old_indexes[index]],dtype=numpy.float32) 
                            rs = self.nodes[index].stamp_sign(x_shr)
                            false_mask_left = zeros((x.shape[0],), dtype=bool)
                            false_mask_left[old_indexes[index]] = rs < 0

                            if false_mask_left.sum():
                                new_indexes[self.structure[index][0]] = false_mask_left

                                if self.structure[index][0] == -1:
                                    final_estimators[index] = false_mask_left

                            false_mask_right = zeros((x.shape[0],), dtype=bool)
                            false_mask_right[old_indexes[index]] = rs >= 0
                            
                            if false_mask_right.sum(): 
                                new_indexes[self.structure[index][1]] = false_mask_right

                                if self.structure[index][1] == -1:
                                    if index in final_estimators:
                                        final_estimators[index] +=  false_mask_right
                                    else:
                                        final_estimators[index] = false_mask_right

                    if len(new_indexes)  > 0:
                        old_indexes = new_indexes
                    else:
                        break

                if stat_only:
                    for idx in final_estimators:
                        res[final_estimators[idx]] = self.nodes[idx].predict_stat(x[final_estimators[idx]])
                else: 
                    if Y is not None:
                        cmp_r = []
                        for idx in final_estimators:
                            res[final_estimators[idx]],cmp_res = self.nodes[idx].predict_proba(x[final_estimators[idx]],Y[final_estimators[idx]],use_weight = use_weight)
                            cmp_r += cmp_res
                        return res, cmp_r    
                    else:    
                        for idx in final_estimators:
                            res[final_estimators[idx]] = self.nodes[idx].predict_proba(x[final_estimators[idx]],use_weight = use_weight)

            return res             
        else:
            return "X type must be scipy.sparse.csr_matrix"   

    def getWeights(self):
        res = []
        for ds in self.nodes:
            res.append(ds.model.coef_)

        return  asarray(res)

    def getCounts(self):
        res = []
        for lidx in self.leaves:
            res.append(self.nodes[lidx].counts)
        return asarray(res)

    def estimateChunkWeights(self, norm, diff, min_nom, uniform=True):
        norm = [self.getCounts()]
        
        norm = numpy.concatenate(norm)
        norm = asarray(norm,float) / norm.sum(axis=1).reshape(-1,1)
        
        cov = numpy.dot(norm,numpy.transpose(norm))
        
        for i in range(cov.shape[0]):
            cov[i,i] = 0
        
        cov = numpy.exp(-cov.sum(axis=1))        
        
        min_nom =  cov.min()       
        max_nom =  cov.max()
        diff = max_nom - min_nom        

        for i,lidx in enumerate(self.leaves):
            if uniform:
                self.nodes[lidx].chunk_weight = 1.
            else:
                counts = asarray(self.nodes[lidx].counts,dtype=float) / self.nodes[lidx].counts.sum()
                row_cov = dot(norm,counts.reshape(-1,1))
                row_cov[i,0] = 0
                counts = numpy.exp(-row_cov.sum())  
                if diff > 0:
                    eps = 1e-6
                    self.nodes[lidx].chunk_weight = (counts - min_nom + eps) / diff
                else:
                    self.nodes[lidx].chunk_weight = 1.0    
                #print ("Set node weight:",i, self.nodes[lidx].chunk_weight)
                
    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_deth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = None, \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10.,intercept_scaling=1.,dropout_low=0., dropout_high=0.9, noise=0., cov_dr=0.,criteria='gini'):
        self.criteria = criteria 
        self.leaves = []
        self.max_deth = max_deth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.kernel = kernel
        self.tol = tol
        #self.seed = seed
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.sample_ratio = sample_ratio

        self.feature_ratio=feature_ratio
        self.processed_counter = 0
        self.old_processed_counter = 0
        self.dual = dual
        self.intercept_scaling = intercept_scaling
        self.dropout_low = dropout_low
        self.dropout_high = dropout_high
        self.noise = noise 

        self.cov_dr = cov_dr
        if seed:
            random.seed(seed)
