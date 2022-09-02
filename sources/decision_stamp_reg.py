'''
Created on 26 марта 2016 г.

@author: keen
'''

from numpy import random
from numpy import zeros
from numpy import ones
from numpy.random import randint
import pickle
import uuid
import traceback
import os

from numpy import sign
from numpy import transpose
from numpy import linalg
from scipy.sparse import csr_matrix
from numpy import int8
from numpy import float64
from numpy import int64
from numpy import exp
from numpy import multiply
from numpy import asarray

from numpy import abs

from numpy import where
from numpy import argsort

from numpy import count_nonzero
from numpy import log

from numpy import bincount
from numpy import nonzero
from scipy.sparse import coo_matrix
from numpy import intersect1d
from copy import deepcopy
from numpy import save
from numpy import float64
from numpy import unique
from numpy import absolute
from numpy import apply_along_axis

import numpy as np

import math
import time

from sklearn.linear_model import SGDClassifier
#from SVM import SVM, polynomial_kernel, gaussian_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import numpy
from sympy.utilities.iterables import multiset_permutations
#import linearSVM

from numpy import isnan
from scipy.sparse.csc import csc_matrix

from sklearn.metrics import accuracy_score
#from thundersvm import *
import subprocess
#from memory_profiler import profile
from sklearn.cluster import KMeans

class DecisionStampReg:
    
    def criteriaMSE(self,pj,x):
        return ((pj - x)*(pj - x)).sum()
    
    def swap_rows(self, mat, a, b):
        a_idx = where(mat.indices == a)
        b_idx = where(mat.indices == b)
        mat.indices[a_idx] = b
        mat.indices[b_idx] = a
        return mat.asformat(mat.format)                                      

    def swap_rows_batch(self, mat, a, b):
        buf = mat[a, :]
        mat[a, :] = mat[b, :]
        mat[b, :] = buf
        return mat  
    
    #@profile
    def convexConcaveOptimization(self,x,Y,sample_weight,samp_counts):
        #random.seed()
        self.counts = numpy.zeros((x.shape[0],))
        if x.shape[0] > 0:
            sample_idx = sample_weight > 0
            sample_idx_ran = asarray(range(x.shape[0]))[sample_idx.reshape(-1)]
            Y_tmp = Y[sample_idx.reshape(-1)]
            x_tmp = csr_matrix(x[sample_idx.reshape(-1)],dtype=np.float32)

            #sample X and Y
            if self.sample_ratio*x.shape[0] > 10:
                #idxs =  random.permutation(x_tmp.shape[0])[:int(x_tmp.shape[0]*self.sample_ratio)]            
                idxs = randint(0, x_tmp.shape[0], int(x_tmp.shape[0]*self.sample_ratio)) #bootstrap
                  
                to_add_cnt = numpy.unique(sample_idx_ran[idxs]) 
                x_ = csr_matrix(x_tmp[idxs],dtype=np.float32)
                Y_ = Y_tmp[idxs]
                    
                diff_y = unique(Y_)
                if diff_y.shape[0] > 1:
                    x_tmp = x_
                    Y_tmp = Y_
                    #print ("sampling shape:",diff_y.shape[0])
            else:
                to_add_cnt = sample_idx_ran

            if not (samp_counts is None): 
                self.counts[to_add_cnt] += 1
            
            def nu(arr):
                return asarray([1 + unique(arr[:,i].data,return_counts=True)[1].shape[0] for i in range(arr.shape[1])])
            
            #nonzero_idxs = unique(x_tmp.nonzero()[1]) 
            counts_p = nu(csc_matrix(x_tmp))
            pos_idx = where(counts_p > 1)[0]
            
            if self.features_weight is not None:
                f_idx = where(self.features_weight > 0)[0]
                pos_idx =list(set(pos_idx).intersection(set(f_idx)))
                fw_size = int(self.features_weight[self.features_weight > 0].shape[0]* self.feature_ratio)
            else:
                fw_size = int(x_tmp.shape[1] * self.feature_ratio)
                if fw_size > pos_idx.shape[0]:
                    fw_size = pos_idx.shape[0]
                #fw_size = int(pos_idx.shape[0] * self.feature_ratio)

            #print("x_tmp:",x_tmp.shape)    
            #print("pos_idx",pos_idx)
                
            self.features_weight = random.permutation(pos_idx)[:fw_size]#.astype(int8)

            if fw_size == 0:
                return 0.

            x_tmp = csr_matrix(x_tmp[:,self.features_weight],dtype=np.float32)

            k = KMeans(n_clusters=2)
            H = k.fit_predict(Y_tmp.reshape(-1,1))*2 - 1    

            deltas = zeros(shape=(H.shape[0]))

            orig_criterion = self.criteriaMSE(Y_tmp[H == -1].mean(),Y_tmp[H == -1]) + self.criteriaMSE(Y_tmp[H == 1].mean(),Y_tmp[H == 1])
            for i in range(H.shape[0]):
                H[i] = - H[i]
                la = Y_tmp[H == -1]
                ra = Y_tmp[H == 1]
                if la.shape[0] == 0 or ra.shape[0] == 0:
                    deltas[i] = Y_tmp.std() * Y_tmp.std() 
                else:    
                    deltas[i] = self.criteriaMSE(la.mean(),la) + self.criteriaMSE(ra.mean(),ra) - orig_criterion
                H[i] = - H[i]

            ratio = 1
            #print("deltas:",deltas)

            dm = deltas.max()
            if deltas.max() == 0:
                deltas = ones(shape=(H.shape[1]))  
            else:
                deltas = (deltas / dm)*ratio 

            try:
                if self.kernel == 'linear':
                    if not self.dual:
                        self.model = SGDClassifier(n_iter_no_change=5,loss='squared_hinge', alpha=1. / (100*self.C), fit_intercept=True, max_iter=self.max_iter, tol=self.tol, eta0=0.5,shuffle=True, learning_rate='adaptive')
                        #self.model = LinearSVC(penalty='l2',dual=self.dual,tol=self.tol,C = self.C,max_iter=self.max_iter)
                        self.model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                    else:  
                        self.model = LinearSVC(penalty='l2',dual=self.dual,tol=self.tol,C = self.C,max_iter=self.max_iter)
                        self.model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                    
                #else:
                if self.kernel == 'polynomial':
                    self.model = SVC(kernel='poly',tol=self.tol,C = self.C,max_iter=self.max_iter,degree=4,gamma=self.gamma)
                    self.model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                else:
                    if self.kernel == 'gaussian':
                        self.model = SVC(kernel='rbf',tol=self.tol,C = self.C,max_iter=self.max_iter,gamma=self.gamma)
                        self.model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                        
                            
            except Exception as exp:
                print (str(exp))
                print (x_tmp.shape)
                print(H)
                print(traceback.format_exc())
                return 0.            

            H = self.stamp_sign(x_tmp, sample = False)
            if Y_tmp[H > 0].shape[0] == 0:
                return 0.
            
            self.p0 = Y_tmp[H > 0].mean()
            self.p1 = Y_tmp[H < 0].mean()
            
            #print (self.p0, self.p1)
            gini_res = self.criteriaMSE(Y_tmp.mean(),Y_tmp) - self.criteriaMSE(self.p0,Y_tmp[H == 1]) - self.criteriaMSE(self.p1,Y_tmp[H == -1])
            #print(gini_res)
            self.counts = [] #numpy.hstack([samp_counts,self.counts]) 
            return gini_res    
#public:

    def __init__(self, features_weight, kernel='linear', \
                 sample_ratio=0.5, feature_ratio=0.5,dual=True,C=100.,tol=0.001,max_iter=1000,gamma=1000.,intercept_scaling=1.,dropout_low=0.1, dropout_high=0.9, balance=True,noise=0.,cov_dr=0., criteria="mse"):
        
        self.tol = tol
        #self.features_weight = deepcopy(features_weight)
        self.C = C
        self.gamma = gamma
        self.sample_ratio = sample_ratio
        self.kernel = kernel
        self.max_iter = max_iter
        self.dual = dual
        self.feature_ratio = feature_ratio
        self.intercept_scaling = intercept_scaling
        self.dropout_low = dropout_low
        self.dropout_high = dropout_high  
        self.balance = balance
        self.noise = noise
        self.cov_dr = cov_dr 
        self.chunk_weight = 1.0 
        self.leaf_id = -1
        self.features_weight = deepcopy(features_weight)
    
    def fit(self, x,Y, sample_weight,counts, instability = 0):
        
        gres = self.convexConcaveOptimization(x,Y,sample_weight,counts)
        sample_weightL = zeros(shape=sample_weight.shape,dtype = int8)
        sample_weightR = zeros(shape=sample_weight.shape,dtype = int8)
        
        self.prob = float(sample_weight.sum()) / sample_weight.shape[1]
        self.instability = instability + 1./ self.prob

        if gres > 0:        
            sign_matrix_full = self.stamp_sign(x)
            sign_matrix = multiply(sample_weight.reshape(-1), sign_matrix_full)
            signs = asarray(sign_matrix)
            colsL = where(signs < 0.0)[0]
            colsR = where(signs > 0.0)[0]
            sample_weightL[0,colsL] = 1       
            sample_weightR[0,colsR] = 1  
            self.probL = float(sample_weightL.sum()) / sample_weight.shape[1]
            self.probR = float(sample_weightR.sum()) / sample_weight.shape[1]
        return gres, sample_weightL, sample_weightR        
    
    def stamp_sign(self,x,sample = True):
        if sample:
            #print("stamp x:", x.shape)
            x = x[:,self.features_weight]
        return sign(self.model.predict(x))

    def predict_proba(self,x,Y = None,sample = True, use_weight = True, get_id=False):
        res = zeros((x.shape[0],1))
        leaf_ids =  zeros((x.shape[0],))
        sgns = self.stamp_sign(x,sample)
        
        eps = 1e-6
        
        if use_weight and self.cov_dr > 0: 
            res[sgns < 0] = self.p0*(self.chunk_weight)
            res[sgns >=0] = self.p1*(self.chunk_weight)
        else:
            res[sgns < 0] = self.p0
            res[sgns >=0] = self.p1

        leaf_ids[sgns < 0] = self.leaf_id
        leaf_ids[sgns >= 0] = self.leaf_id + 1          
        if Y is not None:
            cmp_res = []
            cmp = numpy.argmax(res,axis=1) == Y
            for c in cmp:
                cmp_res.append([int(c),self.chunk_weight,self.counts])
            return res, cmp_res    
        if get_id:
            return res, leaf_ids
        else:    
            return res
