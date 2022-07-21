# coding: utf-8
'''
Created on 27 марта 2016 г.

@author: keen
'''
from scipy.sparse import csr_matrix
import numpy

from utils import *
import time
import os
import pickle
from decision_stamp import DecisionStamp

@prepareProblem
def fit_(tree,sample_weight,addr):
    command(2,-1,mask= int(1).to_bytes(1,byteorder='little') +  int(0).to_bytes(1,byteorder='little') + pickle.dumps(sample_weight),addr=addr) 

class CO2Tree:
    #@profile
    def buildTree(self, bufs):
        id_ = bufs[0][1]
        parent_id = bufs[0][2]
        model = bufs[0][0][0]
        feature_weights = bufs[0][0][1]
        p0 = bufs[0][0][2]
        p1 = bufs[0][0][3]
        side = bufs[0][0][4]
        class_max = bufs[0][0][5]
        
        ds = DecisionStamp(model,feature_weights,class_max,p0,p1)
        
        while len(self.nodes) <= id_:
            self.nodes.append(None)
            
        mid = max(id_,parent_id)
        while len(self.structure) <= mid:
            self.structure.append([-1,-1])
            
        self.nodes[id_] = ds
        if parent_id > -1:
            self.structure[parent_id][side] = id_
        
        if len(bufs) > 1:
            self.buildTree(bufs[1:])   
      
#public
    def printer(self,p):
        try:
            while True:
                print(p.stdout.readline())    
        except:
            print("Target process is killed")        
    
                    
    def fit(self,x,Y, clusterCfg, sample_weight = None, preprocess = False):
        problem = {
                    'kernel': 'linear',
                    'sample_ratio': 1.0,
                    'feature_ratio': 0.5,
                    'dual': self.dual,
                    'C': self.C,
                    'tol': self.tol,
                    'max_iter': self.max_iter,
                    'intercept_scaling': 1.,
                    'dropout_low': self.dropout_low,
                    'dropout_high': self.dropout_high,
                    'balance': self.balance,
                    'criteria': self.criteria,
                    'max_depth':self.max_deth,
                    }
          
        fit_(self,x,Y,problem,clusterCfg,self.res_name,self.addr,preprocess,sample_weight)        
        print ("Construct a tree from the result file")      
        bufs = readResultFile(self.res_name)  
        self.structure = []
        self.nodes = []
        self.buildTree(bufs)
            
        for i,s in enumerate(self.structure):
            if s[0] == -1 or s[1] == -1:
                self.leaves.append(i)                                                      
   
    def predict(self,x, preprocess = False):
        probs = self.predict_proba(x, preprocess)
        
        return numpy.argmax(probs,axis=1)
   
    
    def predict_proba(self,x, preprocess = False, stat_only = False):
        if isinstance(x,csr_matrix):
            res = numpy.zeros((x.shape[0], self.class_max + 1))

            if preprocess:
                x = expandMatrix(x) 
            
            old_indexes = {0:numpy.ones((x.shape[0],), dtype=bool)}

            final_estimators = {}
            
            if hasattr(self, 'nodes') and len(self.nodes) > 0: 
                while True:
                    new_indexes = {}
                    for index in old_indexes:
                        if index > -1:
                            x_shr = csr_matrix(x[old_indexes[index]],dtype=numpy.float32) 
                            rs = self.nodes[index].stamp_sign(x_shr)
                            false_mask_left = numpy.zeros((x.shape[0],), dtype=bool)
                            false_mask_left[old_indexes[index]] = rs < 0
    
                            if false_mask_left.sum():
                                new_indexes[self.structure[index][0]] = false_mask_left
    
                                if self.structure[index][0] == -1:
                                    final_estimators[index] = false_mask_left
    
                            false_mask_right = numpy.zeros((x.shape[0],), dtype=bool)
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
                    for idx in final_estimators:
                        res[final_estimators[idx]] = self.nodes[idx].predict_proba(x[final_estimators[idx]])

            return res             
        else:
            return "X type must be scipy.sparse.csr_matrix"   

    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_depth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = None, \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10.,intercept_scaling=1.,dropout_low=0., dropout_high=0.9, noise=0., cov_dr=0.,criteria='gini'):
        self.criteria = criteria 
        self.leaves = []
        self.max_deth = max_depth
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
        self.intercept_scaling = intercept_scaling
        self.dropout_low = dropout_low
        self.dropout_high = dropout_high
        self.noise = noise 
        self.res_name = "result.bin" 
        self.addr = ("localhost",5555)
        self.balance = True

        self.cov_dr = cov_dr
        if seed:
            numpy.random.seed(seed)
