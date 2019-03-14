# coding: utf-8

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

'''
Created on 27 марта 2016 г.

@author: keen
'''
import CO2_tree as co2
from multiprocessing import Pool
from functools import partial

from numpy import argmax
from numpy import multiply
from numpy import asarray
from joblib import Parallel, delayed
import os
from scipy.sparse.csr import csr_matrix

from numpy import load
from numpy import save
import uuid


def fitter(uuids,forest,shapex,seed_):
    dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
    indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
    ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
    x = csr_matrix((dataX,indX,ptrX), shape=shapex,copy=False)
    Y = load('/dev/shm/' + uuids + "DataY.npy",mmap_mode='r')
        
    tree = co2.CO2Tree(C=forest.C , kernel=forest.kernel,\
    tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_deth,\
     min_samples_split = forest.min_samples_split,dual=forest.dual,\
    min_samples_leaf = forest.min_samples_leaf, seed = seed_,\
     sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
     gamma=forest.gamma)
    tree.fit(x,Y, preprocess = False)
    return tree

       
def probber(uuids, shapex,tree):
        dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
        indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
        ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
        x = csr_matrix((dataX,indX,ptrX), shape=shapex,copy=False)    
        return tree.predict_proba(x,preprocess = False)    

class CO2_forest:
    
    def fit(self,x,Y):
        uuids = str(uuid.uuid4())
        
        save('/dev/shm/'+ uuids + "DataX",x.data)
        save('/dev/shm/'+ uuids + "IndX",x.indices)
        save('/dev/shm/'+ uuids + "PtrX",x.indptr)
        save('/dev/shm/'+ uuids + "DataY",Y) 
        
        #pool = Pool(self.n_jobs)
        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(fitter)(uuids,self,x.shape,i) for i in range(self.n_estimators))
            
        #self.trees = pool.map(func,range(self.n_estimators))            
        #pool.close()
        #pool.join()
        
        os.remove('/dev/shm/'+ uuids + "DataX.npy")
        os.remove('/dev/shm/'+ uuids + "IndX.npy")
        os.remove('/dev/shm/'+ uuids + "PtrX.npy")
        os.remove('/dev/shm/'+ uuids + "DataY.npy")                        

    def predict(self,x):
        proba = self.predict_proba(x)
        return argmax(proba, axis = 1)
    
    def predict_proba(self,x):
        uuids = str(uuid.uuid4())
        
        save('/dev/shm/'+ uuids + "DataX",x.data)
        save('/dev/shm/'+ uuids + "IndX",x.indices)
        save('/dev/shm/'+ uuids + "PtrX",x.indptr)        

        res = Parallel(n_jobs=self.n_jobs)(delayed(probber)(uuids,x.shape,t) for t in self.trees)

        os.remove('/dev/shm/'+ uuids + "DataX.npy")
        os.remove('/dev/shm/'+ uuids + "IndX.npy")
        os.remove('/dev/shm/'+ uuids + "PtrX.npy")

        return multiply(asarray(res).sum(axis=0), 1. / self.n_estimators)
        
    def __init__(self,C, kernel = 'linear', max_deth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,gamma=1000.):
        self.C = C
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.kernel = kernel
        self.max_deth = max_deth
        self.n_estimators = n_estimators 
        self.n_jobs = n_jobs
        self.trees = []
        self.tol = tol
        self.sample_ratio = sample_ratio
        self.gamma = gamma
        self.dual = dual
        self.max_iter = max_iter
        self.feature_ratio=feature_ratio

        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        
        
