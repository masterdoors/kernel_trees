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

from sklearn.tree import DecisionTreeClassifier

from numpy import isnan
from scipy.sparse.csc import csc_matrix

from sklearn.metrics import accuracy_score
#from thundersvm import *
import subprocess
#from memory_profiler import profile

import psutil

class DecisionStamp:
    
    def estimateTetas(self,x,Y,train_data):
        counts = self.n_classes

        self.Teta0 = zeros((counts))
        self.Teta1 = zeros((counts))

        signs = self.stamp_sign(x,train_data,sample = False)

        if isinstance(signs, csr_matrix) or isinstance(signs, coo_matrix):
            signs = signs.todense()

        cl = asarray(multiply(signs,Y + 1))

        cl = cl[nonzero(cl)]

        pos_cl = abs(cl[cl >= 0.0]).astype(int64)
        neg_cl = abs(cl[cl < 0.0]).astype(int64)

        lcount = bincount(neg_cl)
        rcount = bincount(pos_cl)

        for i in range(1,len(lcount)):
                self.Teta0[self.class_map[i - 1]] += float(lcount[i])


        for i in range(1,len(rcount)):
                self.Teta1[self.class_map[i - 1]] += float(rcount[i])

        
    def delta(self,H,Y):
        res = 0
        for s in (-1,+1):
            Hl = float(H[H==s].size) / H.size
            index = asarray(range(H.shape[1]))
            Hs_index = index[H[0,index] == s]
            for y in self.class_map_inv:
                y_index = index[Y[index] == y]
                common_ids = intersect1d(y_index, Hs_index) 
                if Hs_index.shape[0] != 0:
                    pj = float(common_ids.shape[0]) / Hs_index.shape[0]
                    res += Hl * pj * (1 - pj) 
 
        return res       
    
    def criteriaGini(self,pj):
        return pj*(1 - pj)
    
    def criteriaIG(self,pj):
        eps = 0.0000001
        if pj == 0:
            pj += eps
        return - pj*numpy.log(pj)       
    
    def criteriaGinirow(self,pj):
        return (pj*(1 - pj)).sum()
    
    def criteriaIGrow(self,pj):
        eps = 0.0000001
        pj[pj == 0] = eps
        return (- pj*numpy.log(pj)).sum()  


    def getDeltaParams(self,H,Y,criteria):
        res = 0
        IH = {}
        IH[-1] = float(H[H==-1].size)
        IH[1] = float(H[H==+1].size)
        Hsize = H.size
        IY = {}
        IY[-1] = {}
        IY[1] = {}
        
        for s in (-1,+1):
            Hl = float(H[H==s].size) / H.size
                
            index = asarray(range(H.shape[1]))
            Hs_index = index[H[0,index] == s]
            
            for y in self.class_map_inv:
                y_index = index[Y[index] == y]
                common_ids = intersect1d(y_index, Hs_index) 
                IY[s][y] = common_ids.shape[0]

                if Hs_index.shape[0] != 0:
                    pj = float(common_ids.shape[0]) / Hs_index.shape[0]
                    res += Hl *  criteria(pj) 
 
        return Hsize, IH,IY, res           
    
    def delta_wise(self, Hsize, IH,IY,yi,hi, criteria):
        res = 0
        for s in [-1,1]:
            Hl = float(IH[s] + hi*s) / Hsize
            if IH[s] + hi*s > 0:
                for y in self.class_map_inv:
                    pj = 0
                    if y == yi:
                        pj = (IY[s][y] + hi*s) / (IH[s] + hi*s)
                    else:    
                        pj = (IY[s][y]) / (IH[s] + hi*s)
                    
                    res += Hl * criteria(pj)
            if math.isnan(res):
                res = 0
        return res              

    def calcGiniInc(self,w,x,Y):
        signs = transpose(sign(x - w))
       
        cl = asarray(multiply(signs,Y))
        
        cl = cl[nonzero(cl)]
        
        pos_cl = abs(cl[cl >= 0.0]).astype(int64)
        neg_cl = abs(cl[cl < 0.0]).astype(int64)
        cl = abs(cl).astype(int64)
            
        gl = 0
        gr = 0
        ga = 0
        
        lcount = bincount(neg_cl)
        rcount = bincount(pos_cl)
        acount = bincount(cl)    

        for l in lcount:
            gl += (float(l)/len(neg_cl)) * (1 - float(l)/len(neg_cl))

        for r in rcount:
            gr += (float(r)/len(pos_cl)) * (1 - float(r)/len(pos_cl))
         
        for a in acount:
            ga += (float(a)/len(cl)) * (1 - float(a)/len(cl))
            
        if len(cl) > 0:     
            return  ga - (float(len(neg_cl)) / len(cl))*gl -(float(len(pos_cl)) / len(cl))*gr
        else:
            return 0.        
            
    def calcGini(self,x,Y, train_data,report = False):
        signs = self.stamp_sign(x,train_data)
        
        if isinstance(signs, csr_matrix) or isinstance(signs, coo_matrix): 
            signs = signs.todense()        
       
        cl = asarray(multiply(signs,Y))
        
        if isnan(cl).any():
            return 0.0

        cl = cl[nonzero(cl)]
        
        pos_cl = abs(cl[cl >= 0.0]).astype(int64)
        neg_cl = abs(cl[cl < 0.0]).astype(int64)
        cl = abs(cl).astype(int64)
            
        gl = 0
        gr = 0
        ga = 0
        
        lcount = bincount(neg_cl)
        rcount = bincount(pos_cl)
        acount = bincount(cl)
        
        if report:
            #print "Classes:",  acount
            #print "Classes: ", lcount
            #print "Classes: ", rcount
            #print " (",lcount.sum(),",",rcount.sum(),")"
            pass

        for l in lcount:
            gl += (float(l)/len(neg_cl)) * (1 - float(l)/len(neg_cl))

        for r in rcount:
            gr += (float(r)/len(pos_cl)) * (1 - float(r)/len(pos_cl))
         
        for a in acount:
            ga += (float(a)/len(cl)) * (1 - float(a)/len(cl))
            
        #print "Impurities: ", ga, gl, gr 
        if len(cl) > 0:     
            return  ga - (float(len(neg_cl)) / len(cl))*gl -(float(len(pos_cl)) / len(cl))*gr
        else:
            return 0.
        
    def bijection(self,w,x,i,Y,minv,maxv,eps):
        if maxv - minv > eps:
            midllev = (maxv + minv) / 2
            w_ = midllev - (midllev - minv) /2  
            l = self.calcGiniInc(w_,x,Y)              
            w_ = midllev + (maxv - midllev) /2    
            r = self.calcGiniInc(w_,x,Y)               
            if l > r:
                return self.bijection(w,x,i,Y,minv,midllev,eps) 
            else:
                return self.bijection(w,x,i,Y,midllev,maxv,eps)                     
        else:
            w.fill(0.0)
            w[0,i] = 1.0
            w[0,w.shape[1] - 1] = minv  
            return self.calcGiniInc(minv,x,Y)         
    
    
    def searchDivider(self,w,x,Y,i):
        eps = 0.0000001
        w.fill(0.0)
        x_ = asarray(x.getcol(i).todense())
        maxv = x_.max(axis = 0)[0]
        minv = x_.min(axis = 0)[0]
        
        if maxv - minv > eps:        
            return  self.bijection(w,x_,i,Y,minv,maxv,eps) 
        else:
            return 0.0 
    
    def genBestSplit(self,x,Y,q):
        
        fw = self.features_weight.todense()
        min_impurity = 0
        min_impurity_w = zeros((1, x.shape[1]))
        
        indexies = asarray(argsort(fw))[0,:]
        
        counter = self.features_weight.shape[1] - self.features_weight.nnz
        
        qm = counter + len(bincount(Y))
        
        while (counter < qm or min_impurity == 0) and counter < x.shape[1] - 1:

            i = randint(counter,x.shape[1] - 1)
            w_ = zeros((1, x.shape[1]))
            imp = self.searchDivider(w_,x,Y, indexies[i])
               
            if imp > min_impurity:
                min_impurity = imp
                min_impurity_w = w_
            
            if imp == 0:
                fw[0, indexies[i]] = 0.0    
                
            t = indexies[counter]
            indexies[counter] = indexies[i]
            indexies[i] = t                 
                
            counter += 1    
            
        self.features_weight = csr_matrix(fw)    
  
        return min_impurity_w

    def genBestSplitSerial(self,x,Y,q):
        
        fw = self.features_weight.todense()
        min_impurity = 0
        
        indexies = asarray(argsort(fw))[0,:]
        
        counter = self.features_weight.shape[1] - self.features_weight.nnz
        
        qm = counter + q
        
        imp_list = []
        
        while (counter < qm or min_impurity == 0) and counter < x.shape[1] - 1:

            i = randint(counter,x.shape[1] - 1)
            w_ = zeros((1, x.shape[1]))
            imp = self.searchDivider(w_,x,Y, indexies[i])
               
            if imp > min_impurity:
                min_impurity = imp
                imp_list.append(w_)
            
            if imp == 0:
                fw[0, indexies[i]] = 0.0    
                
            t = indexies[counter]
            indexies[counter] = indexies[i]
            indexies[i] = t                 
                
            counter += 1    
            
        self.features_weight = csr_matrix(fw)    
  
        return imp_list, min_impurity
    
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
            #x_tmp = csr_matrix(x[sample_idx.reshape(-1)],dtype=np.float32)
            rng = random.default_rng(self.seed)#+abs(int(x_tmp.sum())))

            #sample X and Y
            if self.sample_ratio*x.shape[0] > 10:
                #idxs =  random.permutation(x_tmp.shape[0])[:int(x_tmp.shape[0]*self.sample_ratio)]   
                idxs = rng.integers(0, sample_idx_ran.shape[0], int(sample_idx_ran.shape[0]*self.sample_ratio)) #bootstrap
                  
                to_add_cnt = numpy.unique(sample_idx_ran[idxs]) 
                #x_ = csr_matrix(x_tmp[idxs],dtype=np.float32)
                Y_ = Y_tmp[idxs]
                    
                diff_y = unique(Y_)
                if diff_y.shape[0] > 1:
                    #x_tmp = x_
                    sample_idx_ran = sample_idx_ran[idxs]
                    Y_tmp = Y_
                    #print ("sampling shape:",diff_y.shape[0])
            else:
                to_add_cnt = sample_idx_ran

            if not (samp_counts is None): 
                self.counts[to_add_cnt] += 1
            
            def nu(arr):
                return asarray([1 + unique(arr[:,i].data,return_counts=True)[1].shape[0] for i in range(arr.shape[1])])
            
            #nonzero_idxs = unique(x_tmp.nonzero()[1]) 
            counts_p = nu(csc_matrix(x[sample_idx_ran]))
            pos_idx = where(counts_p > 1)[0]
            
            if self.features_weight is not None:
                f_idx = where(self.features_weight > 0)[0]
                pos_idx =list(set(pos_idx).intersection(set(f_idx)))
                fw_size = int(self.features_weight[self.features_weight > 0].shape[0]* self.feature_ratio)
            else:
                fw_size = int(x.shape[1] * self.feature_ratio)
                if fw_size > pos_idx.shape[0]:
                    fw_size = pos_idx.shape[0]
                #fw_size = int(pos_idx.shape[0] * self.feature_ratio)

            self.features_weight = rng.permutation(pos_idx)[:fw_size]#.astype(int8)

            if fw_size == 0:
                return 0.
            self.sample_weight = sample_idx_ran 
            #x_tmp = csr_matrix(x_tmp[:,self.features_weight],dtype=np.float32)
            
            #print (x_tmp.shape,fw_size)

            H = zeros(shape = (1,Y_tmp.shape[0]))        
            
            gini_res = 0    
    
            class_counts = unique(Y_tmp, return_counts=True)
            class_counts = numpy.asarray(list(zip(class_counts[0],class_counts[1])))

            class2side = {}
            class2count = {}
            side2count = {}

            min_gini = self.max_criteria
            min_p = []
            
            if len(class_counts) > 13:
            #Greedy
                for _ in range(len(class_counts)*len(class_counts)):
                    lmin_gini = self.max_criteria
                    lmin_p = []

                    next = True
                    elements = [-1,+1]
                    probabilities = [0.5, 0.5]
                    p = numpy.random.choice(elements,len(class_counts) , p=probabilities)

                    zc = 0 
                    while next:
                        next = False
                        zc += 1  
                        for i in range(p.shape[0]):
                            p[i] = - p[i]
                            left_counts = class_counts[p < 0, 1]
                            right_counts = class_counts[p > 0, 1]

                            lcs = left_counts.sum()
                            rcs = right_counts.sum()  
                            den = lcs + rcs 

                            PL = float(lcs)/ den
                            PR = float(rcs)/ den
            
                            gini_l = self.criteria_row(left_counts / lcs)
                            gini_r = self.criteria_row(right_counts / rcs)

                            gini =  PL*gini_l + PR* gini_r
                            if gini < lmin_gini:
                                lmin_p = deepcopy(p)
                                lmin_gini = gini
                                next = True
                        p = lmin_p

                    if  lmin_gini < min_gini:
                        min_p = deepcopy(lmin_p)
                        min_gini = lmin_gini
 
            else:
                for zc in range(1,len(class_counts),1):
                    a = numpy.hstack([-numpy.ones((zc,)),numpy.ones((len(class_counts) - zc,))])
                    for p in multiset_permutations(a):
                        p = numpy.asarray(p)
                        left_counts = class_counts[p < 0, 1]
                        right_counts = class_counts[p > 0, 1]

                        lcs = left_counts.sum()
                        rcs = right_counts.sum()  
                        den = lcs + rcs 

                        PL = float(lcs)/ den
                        PR = float(rcs)/ den
            
                        gini_l = self.criteria_row(left_counts / lcs)
                        gini_r = self.criteria_row(right_counts / rcs)

                        gini =  PL*gini_l + PR* gini_r

                        if gini < min_gini:
                            min_p = p
                            min_gini = gini

            left_counts = numpy.asarray([c[1] for c in class_counts[min_p < 0]])
            right_counts = numpy.asarray([c[1] for c in class_counts[min_p > 0]])
            side2count[-1] = left_counts.sum()
            side2count[1] = right_counts.sum()               
            for i,(cl,cnt) in enumerate(class_counts):
                class2side[cl] = min_p[i]
                H[0,Y_tmp == cl] = min_p[i]     
                class2count[cl] = cnt

            gini_best = 0
            gini_old = 0
            for class_id, count_ in class_counts:
                p = float(count_) / side2count[class2side[class_id]]
                p2 = float(count_) / (side2count[-1] + side2count[1])

                gini_old += self.criteria(p2)
                gini_best +=  (float(side2count[class2side[class_id]])/ (side2count[-1] + side2count[1]))*self.criteria(p)

            Hsize, IH,IY, gini_old_wise = self.getDeltaParams(H,Y_tmp, self.criteria)

            gini_best = gini_old - gini_best

            deltas = zeros(shape=(H.shape[1]))
            #deltas = ones(shape=(H.shape[1])) 
            for i in range(H.shape[1]):
                gini_i = self.delta_wise(Hsize, IH,IY,Y_tmp[i],-H[0,i],self.criteria)
                deltas[i] = float(gini_i - gini_old_wise)  

                if self.balance:
                    deltas[i] = deltas[i] * float(H.reshape(-1).shape[0]) / (2*side2count[H[0,i]])

            #deltas = deltas - deltas.min()

            ratio = 1

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
                        self.model.fit(x[sample_idx_ran][:, self.features_weight],H.reshape(-1),sample_weight=deltas)
                    else:  
                        self.model = LinearSVC(penalty='l2',dual=self.dual,tol=self.tol,C = self.C,max_iter=self.max_iter)
                        self.model.fit(x[sample_idx_ran][:, self.features_weight],H.reshape(-1),sample_weight=deltas)
                    
                #else:
                if self.kernel == 'polynomial':
                    self.model = SVC(kernel='poly',tol=self.tol,C = self.C,max_iter=self.max_iter,degree=4,gamma=self.gamma)
                    self.model.fit(x,H.reshape(-1),self.features_weight,sample_idx_ran,sample_weight=deltas)  
                else:
                    if self.kernel == 'gaussian':
                        self.model = SVC(kernel='rbf',tol=self.tol,C = self.C,max_iter=self.max_iter,gamma=self.gamma,cache_size=4000)
                        self.model.fit(x,H.reshape(-1),self.features_weight,sample_idx_ran,sample_weight=deltas)   
                        out = self.model.predict(x[:, self.features_weight],x[sample_idx_ran][:, self.features_weight])
                    else:
                        if self.kernel == 'univariate':
                            self.model = DecisionTreeClassifier( criterion= self.criteria_str, max_depth=1)
                            self.model.fit(x[sample_idx_ran][:, self.features_weight],H.reshape(-1))
                        
            except Exception as exp:
                print (str(exp))
                #print (x_tmp.shape)
                print(H)
                print(traceback.format_exc())
                return 0.            

            print("Before Gini: ", Y_tmp.shape,self.features_weight.shape,psutil.virtual_memory())
            
            gini_res = self.calcGini(x[sample_idx_ran],Y_tmp,x)
            
            print("After Gini: ", Y_tmp.shape,self.features_weight.shape,psutil.virtual_memory())

            #print (gini_res,"|",gini_best)

            self.estimateTetas(x[sample_idx_ran][:, self.features_weight], Y_tmp,x) 

            self.p0 = zeros(shape=(self.class_max + 1))
            self.p1 = zeros(shape=(self.class_max + 1))

            sum_t0 = self.Teta0.sum()
            sum_t1 = self.Teta1.sum()

            if sum_t0 > 0: 
                p0_ = multiply(self.Teta0, 1. / sum_t0)                

                for i in range(len(p0_)):
                    self.p0[self.class_map_inv[i]] = p0_[i]

                #exp_0 = exp(self.p0)
                #self.p0 = exp_0 / exp_0.sum()

            if sum_t1 > 0:       
                p1_ = multiply(self.Teta1, 1. / sum_t1)

                for i in range(len(p1_)):
                    self.p1[self.class_map_inv[i]] = p1_[i]  

                #exp_1 = exp(self.p1)
                #self.p1 = exp_1 / exp_1.sum()
            self.counts = [] #numpy.hstack([samp_counts,self.counts]) 
            #print ("pass 3")
            return gini_res    
#public:

    def __init__(self, n_classes,class_max, features_weight, kernel='linear', \
                 sample_ratio=0.5, feature_ratio=0.5,dual=True,C=100.,tol=0.001,max_iter=1000,gamma=1000.,intercept_scaling=1.,dropout_low=0.1, dropout_high=0.9, balance=True,noise=0.,cov_dr=0., criteria="gini",seed=None):
        
        coin = randint(0,2)
        
        if criteria == "gain":
            self.criteria_str = 'entropy'
            self.criteria = self.criteriaIG
            self.criteria_row = self.criteriaIGrow
            self.max_criteria = 1e32
        else:
            self.criteria_str = criteria
            self.criteria = self.criteriaGini 
            self.criteria_row = self.criteriaGinirow
            self.max_criteria = 1.0       
        
        self.tol = tol
        self.n_classes = n_classes
        self.class_max = class_max
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
        if seed is not None:
            self.seed = seed
    
    def fit(self, x,Y, sample_weight,class_map,class_map_inv,counts, instability = 0):
        
        self.class_map = class_map
        self.class_map_inv = class_map_inv
        
        gres = self.convexConcaveOptimization(x,Y,sample_weight,counts)
        #x = deepcopy(x)
        sample_weightL = zeros(shape=sample_weight.shape,dtype = int8)
        sample_weightR = zeros(shape=sample_weight.shape,dtype = int8)
        
        self.prob = float(sample_weight.sum()) / sample_weight.shape[1]
        self.instability = instability + 1./ self.prob

        if gres > 0:        
            sign_matrix_full = self.stamp_sign(x,x)
            sign_matrix = multiply(sample_weight.reshape(-1), sign_matrix_full)
            signs = asarray(sign_matrix)
            colsL = where(signs < 0.0)[0]
            colsR = where(signs > 0.0)[0]
            sample_weightL[0,colsL] = 1       
            sample_weightR[0,colsR] = 1  
            self.probL = float(sample_weightL.sum()) / sample_weight.shape[1]
            self.probR = float(sample_weightR.sum()) / sample_weight.shape[1]
            #print (sample_weightL.shape,sample_weightR.shape)
        #print ("pass 4")
        return gres, sample_weightL, sample_weightR        
    
    def stamp_sign(self,x,train_data, sample = True):
        cur_id = str(uuid.uuid4())     
        
        if sample:
            if self.kernel == 'linear' or self.kernel == 'univariate':
                return sign(self.model.predict(x[:,self.features_weight]))
            else:  
                res = self.model.predict(x[:,self.features_weight], train_data[self.sample_weight][:,self.features_weight])
                return sign(res)
        else:
            if self.kernel == 'linear' or self.kernel == 'univariate':
                return sign(self.model.predict(x))
            else:  
                res = self.model.predict(x, train_data[self.sample_weight][:,self.features_weight])
                return sign(res)            

    def predict_stat(self,x,sample = True):
        res = zeros((x.shape[0],self.class_max + 1))

        sgns = self.stamp_sign(x,sample)

        res[sgns < 0,1:] = self.Teta0
        res[sgns >=0,1:] = self.Teta1

        return res

    def predict_proba(self,x,Y = None,train_data=None,sample = True, use_weight = True, get_id=False):
        res = zeros((x.shape[0],self.class_max + 1))
        leaf_ids =  zeros((x.shape[0],))
        sgns = self.stamp_sign(x, train_data, sample)

        #print("chunk weight orig: ",self.p0, self.p1)
        #print("chunk weight: ",self.p0*self.chunk_weight, self.p1*self.chunk_weight) 
        
        eps = 1e-6
        #self.cov_dr
        #print ("Use weight:", self.chunk_weight, self.p0,self.p1)
        
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
