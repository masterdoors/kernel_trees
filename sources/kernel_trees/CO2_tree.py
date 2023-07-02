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
import collections

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
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

import numpy

import sys
from sklearn.preprocessing import LabelEncoder


def expandMatrix(x):
        x = csr_matrix((x.data, x.indices, x.indptr),shape=(x.shape[0], x.shape[1] + 1), dtype = numpy.float32, copy=False)
        tdat = [-1] * x.shape[0]
        tcols = [x.shape[1] - 1] *  x.shape[0]
        trows = range(x.shape[0])
        
        x_tmp = csr_matrix((tdat, (trows, tcols)),shape=x.shape,dtype = numpy.float32)
        
        x = x + x_tmp     
        
        return x

class BaseCO2Tree:
    
    def clearNode(self,Y,sample_weight):
        y = asarray(multiply(sample_weight.reshape(-1),Y))
        
        y = y[nonzero(y)]
                
        differents = unique(y).shape[0]
        
        if differents <= 1:
            return True
        else:
            return False        

    #@profile
    def buildTree(self, tol, C, x, Y, structure, sample_weight, features_weight, deth,balanced,sam_counts,instability=0):
        assert self.decisionStampClass is not None
        if self.processed_counter > 0:
            if self.old_processed_counter != self.processed_counter:
                #print "Already processed: ", self.processed_counter
                self.old_processed_counter = self.processed_counter
        nnz = count_nonzero(sample_weight)

        if  self.max_depth is None or deth <= self.max_depth: 
            if nnz >= self.min_samples_split: 
                if not self.clearNode(Y, sample_weight):
                    #print "deth:", deth
                    #balanced = True
                    cf = 1

                    #if deth < 3:
                    #    sample_ratio = 0.2 * self.sample_ratio
                    #else:
                    sample_ratio = self.sample_ratio

                    
                    ds = self.decisionStampClass(self.n_classes,self.class_max, features_weight,\
                                           self.kernel, sample_ratio,self.feature_ratio,\
                                           self.dual,C/cf,tol,self.max_iter,self.gamma,balanced,\
                                           self.criteria,seed=self.seed, verbose = self.verbose )

                    gres,sample_weightL, sample_weightR = ds.fit(x, Y, sample_weight,self.class_map,self.class_map_inv,sam_counts,instability)
                    #print ("R:",gres,sample_weightL.shape[0], sample_weightR.shape[0])
    
                    
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
                        #print ("pass 5")
                        if nnzL > self.min_samples_leaf:      
                            #left
                            if not (sam_counts is None): 
                                structure[id_][0] = self.buildTree(tol, C, x, Y, structure, sample_weightL, features_weight, deth + 1,balanced,deepcopy(ds.counts),ds.instability)
                            else:
                                structure[id_][0] = self.buildTree(tol, C, x, Y, structure, sample_weightL, features_weight, deth + 1,balanced,sam_counts,ds.instability)    
                        else:
                            self.processed_counter += nnzL                                
                            #right
                        if nnzR > self.min_samples_leaf:    
                            if not (sam_counts is None):    
                                structure[id_][1] = self.buildTree(tol, C, x, Y, structure, sample_weightR, features_weight, deth + 1,balanced,deepcopy(ds.counts),ds.instability)
                            else:
                                structure[id_][1] = self.buildTree(tol, C, x, Y, structure, sample_weightR, features_weight, deth + 1,balanced,sam_counts,ds.instability)                                    
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
    #@profile
    def fit(self,x,Y, sample_weight = None, preprocess = False):
        if (isinstance(x,csr_matrix) or isinstance(x,numpy.ndarray)) and isinstance(Y,ndarray):
            if Y.shape[0] > 0 and x.shape[0] == Y.shape[0]:
                
                if preprocess:
                    x = expandMatrix(x)
                    self.le = LabelEncoder().fit(Y)
                    Y = self.le.transform(Y)
                
                self.n_features = x.shape[1]
                
                if Y.dtype == int64 or Y.dtype == int8: 
                    classes_ = nonzero(bincount(Y))[0]
                    self.n_classes = len(classes_)
                    self.class_max = Y.max()
                    
                    self.class_map = {}#zeros(shape = (self.class_max + 1), dtype = int64)
                    self.class_map_inv ={} #zeros(shape = (self.n_classes), dtype = int64)
                    cc = 0
                    for c in classes_:
                        self.class_map[c] = cc
                        self.class_map_inv[cc] = c                   
                        cc += 1 
                else:
                    self.n_classes = 1
                    self.class_max = 0     
                    self.class_map = {}    
                    self.class_map_inv ={}   
                
                if sample_weight == None:
                    sample_weight = ones(shape=(1,x.shape[0]),dtype = int8)
                self.nodes = []

                if self.spatial_mul < 1.:
                    features_weight = numpy.zeros((x.shape[1],))
                    m = int(numpy.sqrt(x.shape[1]))
                    i = numpy.random.randint(0,int((1 - self.spatial_mul)*m))
                    j = numpy.random.randint(0,int((1 - self.spatial_mul)*m))
                    for i_ in range(i,i + int(self.spatial_mul*m)):
                        for j_ in range(j,j + int(self.spatial_mul*m)):
                            num = i_*(m-1) + j_
                            features_weight[num] = 1
                else:
                    features_weight = None
                
                self.structure = []
                
                sam_counts = None
                        
                self.total_len = self.buildTree(self.tol, self.C,x, Y, self.structure, sample_weight, features_weight, 1,True,sam_counts)
                

                leaf_id = 0
                self.adj_leaves = {}
                for i,s in enumerate(self.structure):
                    if s[0] == -1 or s[1] == -1:
                        self.leaves.append(i) 
                        self.nodes[i].leaf_id = leaf_id
                        if s[0] + s[1] == -2:
                            leaf_id += 2
                            self.adj_leaves[leaf_id] = [self.nodes[i].instability,1./(self.nodes[i].probL + 0.0001),1./(self.nodes[i].probR + 0.0001)]                            
                        else:
                            leaf_id += 1
                
                #print ("Tree is ready")
                 
            else:
                print ("Wrong training set dimensionality")  
        else:
            print ("X type must be scipy.sparse.csr_matrix and Y type must be numpy.ndarray")          

    
    def predict(self,x, train_data, preprocess = False):
        probs = self.predict_proba(x, train_data = train_data, preprocess = preprocess)
        if self.le is not None:
            return self.le.inverse_transform(argmax(probs,axis=1))
        else:
            return probs
   
    
    def predict_proba(self,x, Y = None,train_data = None, preprocess = False, stat_only = False, use_weight = True, return_leaf_id = False):
        if isinstance(x,csr_matrix) or isinstance(x,numpy.ndarray):
            res = zeros((x.shape[0], self.n_classes))
            leaf_ids = zeros((x.shape[0],))
            
            if self.total_len > -1:
            
                if preprocess:
                    x = expandMatrix(x) 
                
                old_indexes = {0:ones((x.shape[0],), dtype=bool)}

                final_estimators = {}

                while True:
                    new_indexes = {}
                    for index in old_indexes:
                        if index > -1:
                            #
                            x_shr = x[old_indexes[index]] 
                            rs = self.nodes[index].stamp_sign(x_shr, train_data)
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
                            res[final_estimators[idx]],cmp_res = self.nodes[idx].predict_proba(x[final_estimators[idx]],Y[final_estimators[idx]],train_data=train_data,use_weight = use_weight)
                            cmp_r += cmp_res
                        return res, cmp_r    
                    else:    
                        for idx in final_estimators:
                            if return_leaf_id:
                                res[final_estimators[idx]], leaf_ids[final_estimators[idx]] =  self.nodes[idx].predict_proba(x[final_estimators[idx]],train_data=train_data,use_weight = use_weight, get_id = True)  
                            else:    
                                res[final_estimators[idx]] = self.nodes[idx].predict_proba(x[final_estimators[idx]],train_data=train_data,use_weight = use_weight)

            if return_leaf_id:
                return res,leaf_ids 
            else:
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
    
    def getProbs(self):
        res = numpy.zeros((pow(2,self.max_depth-1),))
        for i,lidx in enumerate(self.leaves):
            res[i] = self.nodes[lidx].prob
        return res  
    
    def getIndicators(self,x, train_data = None,noise = 0., balance_noise = False):
        _, lids = self.predict_proba(x, Y = None,train_data=train_data,preprocess = False, stat_only = False, use_weight = False,return_leaf_id = True)
    
        self.leaves_number = 0
        for s in self.structure:
            if s[0] == -1:
                self.leaves_number += 1
            if s[1] == -1:
                self.leaves_number += 1
        indicators = numpy.zeros((lids.shape[0],self.leaves_number))    
        for i in range(lids.shape[0]):
            if int(lids[i]) < self.leaves_number:
                indicators[i,int(lids[i])] = 1.  
        
        if noise > 0:
            if balance_noise:
                population = indicators.sum(axis=0).max()
                for i in range(indicators.shape[1]):
                    dec_ratio = float(indicators[:,i].sum()) / population #balanced noise
                    #print ("Noise balancing. Population: ", population, " indicators: ", indicators.sum(axis=0), "ind_cur:",float(indicators[:,i].sum()),"dec_ratio:", dec_ratio)
                    nonzero = numpy.where(indicators[:,i] > 0)[0]
                    idxs = numpy.random.randint(0, nonzero.shape[0], int(nonzero.shape[0]*noise*dec_ratio)) 
                    indicators[nonzero[idxs],i] = 0. #let's make the classifiers different again                
            else:    
                for i in range(indicators.shape[1]):
                    nonzero = numpy.where(indicators[:,i] > 0)[0]
                    idxs = numpy.random.randint(0, nonzero.shape[0], int(nonzero.shape[0]*noise)) 
                    indicators[nonzero[idxs],i] = 0. #let's make the classifiers different again
        return indicators, self.seed    
            

    def estimateChunkWeights(self, w):
        if isinstance(w, collections.Iterable):
            for _,lidx in enumerate(self.leaves):
                self.nodes[lidx].chunk_weight = 1
                self.nodes[lidx].p0 = w[self.nodes[lidx].leaf_id]
                self.nodes[lidx].p1 = w[self.nodes[lidx].leaf_id + 1]
        else:
            for _,lidx in enumerate(self.leaves):
                self.nodes[lidx].chunk_weight = w
            
                
    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_depth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = 0, \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10.,criteria='gini', spatial_mul=1.0, verbose = 0):
        self.criteria = criteria 
        self.leaves = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.kernel = kernel
        self.tol = tol
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.sample_ratio = sample_ratio

        self.feature_ratio=feature_ratio
        self.processed_counter = 0
        self.old_processed_counter = 0
        self.dual = dual
        self.leaves_number = 0
        self.spatial_mul = spatial_mul
        self.seed = seed
        self.verbose = verbose

        
class CO2TreeClassifier(BaseCO2Tree, ClassifierMixin):
    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_depth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = 0, \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10.,\
                 criteria='gini', spatial_mul=1.0,verbose = 0):
        super().__init__(C, tol, max_iter,kernel, dual,max_depth, \
                 min_samples_split, min_samples_leaf, seed, \
                 sample_ratio,feature_ratio,gamma, \
                 criteria, spatial_mul,verbose)
        self.decisionStampClass = dst.DecisionStampClassifier

class CO2TreeRegressor(BaseCO2Tree, RegressorMixin):
    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_depth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = 0, \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10., \
                criteria='mse', spatial_mul=1.0,verbose = 0):
        super().__init__(C, tol, max_iter,kernel, dual,max_depth, \
                 min_samples_split, min_samples_leaf, seed, \
                 sample_ratio,feature_ratio,gamma, \
                 criteria, spatial_mul,verbose)
        self.decisionStampClass = dst.DecisionStampRegressor


