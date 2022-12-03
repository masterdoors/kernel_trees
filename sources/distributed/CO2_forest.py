# coding: utf-8

import os
import datetime
#os.environ["OPENBLAS_NUM_THREADS"] = "1"

'''
Created on 27 марта 2016 г.

@author: keen
'''
from CO2_tree import prepareProblem
from CO2_tree import CO2Tree

from numpy import argmax
from numpy import multiply
from numpy import asarray

from joblib import Parallel, delayed
from scipy.sparse.csr import csr_matrix


from numpy import load
from numpy import save
import uuid
import numpy
from utils import readResultFile
from utils import BaseCmd,Cmd 
from utils import loadClusterCfg
import pickle

import time
       
def probber(uuids, shapex,tree,stat_only):
        dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
        indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
        ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
        x = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32, copy=False)    
        return tree.predict_proba(x,preprocess = False,stat_only=stat_only)    

@prepareProblem
def fitter_(self,sample_weight,addr):
    
    self.trees = []
    for _ in range(self.n_estimators):
        tree = CO2Tree(C=self.C , kernel=self.kernel,\
        tol=self.tol, max_iter=self.max_iter,max_depth = self.max_deth,\
        min_samples_split = self.min_samples_split,dual=self.dual,\
        min_samples_leaf = self.min_samples_leaf, seed = None,\
        sample_ratio = self.sample_ratio, feature_ratio = self.feature_ratio, \
        gamma=self.gamma,intercept_scaling=self.intercept_scaling,dropout_low=self.dropout_low,dropout_high=self.dropout_high,noise=self.noise,cov_dr=self.cov_dr, criteria = self.criteria)
        self.trees.append(tree)
    
    for _ in self.trees:
        #command(2,-1,mask= int(1).to_bytes(1,byteorder='little') +  int(0).to_bytes(1,byteorder='little') + pickle.dumps(sample_weight),addr=addr)
        Cmd(2,int(1).to_bytes(1,byteorder='little') +  int(0).to_bytes(1,byteorder='little') + pickle.dumps(sample_weight),self.db,self.res)

class CO2_forest:

    def tree_split(self,bufs):
        bufs = sorted(bufs, key=lambda tup: tup[2])
        sep_bufs = []
        id2arr = {}
        for b in bufs:
            id_ = b[1]
            parent_id = b[2]
            if parent_id == -1:
                v = []
                v.append(b)
                sep_bufs.append(v)
                id2arr[id_] = v
            else:
                id2arr[id_] = id2arr[parent_id]
                id2arr[parent_id].append(b)     
        return sep_bufs
    
    #@profile
    def fit(self,x,Y):
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
          
        fitter_(self,x,Y,problem,self.cluster_cgf,self.res_name,self.addr)  
        
        for t in self.trees:
            t.class_max = self.class_max

        print ("Construct a forest from the result file")      
        bufs = readResultFile(self.res_name)    
        spl_res = self.tree_split(bufs)
        for i,buf in enumerate(spl_res):
            self.trees[i].structure = []
            self.trees[i].nodes = []
            self.trees[i].buildTree(buf)
            
            offs2id = {}
            offs = 0
            to_remove = []
            for j in range(len(self.trees[i].nodes)):
                if self.trees[i].nodes[j] is None:
                    to_remove.append(j - offs)
                    offs += 1
                offs2id[j] = j - offs 

            for j in to_remove: 
                    self.trees[i].nodes.pop(j)
                    self.trees[i].structure.pop(j)
                    
            offs2id[-1] = -1        
                
            for j in range(len(self.trees[i].structure)):
                s = self.trees[i].structure[j]
                self.trees[i].structure[j] = [offs2id[s[0]],offs2id[s[1]]]
                        
            for j,s in enumerate(self.trees[i].structure):
                if s[0] == -1 or s[1] == -1:
                    self.trees[i].leaves.append(j)
            print("Tree",i," with ", len(self.trees[i].leaves)," leaves")         
                            
    def predict(self,x):
        proba = self.predict_proba(x)    
        res =  argmax(proba, axis = 1)
        zr = res == 0
        res[zr] = 1
        return res         
    
    def predict_proba(self,x,Y=None,avg='macro'):
        uuids = str(uuid.uuid4())
        
        save('/dev/shm/'+ uuids + "DataX",x.data)
        save('/dev/shm/'+ uuids + "IndX",x.indices)
        save('/dev/shm/'+ uuids + "PtrX",x.indptr)        

        if Y is not None:
            save('/dev/shm/'+ uuids + "DataY",Y) 
        res = Parallel(n_jobs=self.n_jobs)(delayed(probber)(uuids,x.shape,t,False) for t in self.trees)

        os.remove('/dev/shm/'+ uuids + "DataX.npy")
        os.remove('/dev/shm/'+ uuids + "IndX.npy")
        os.remove('/dev/shm/'+ uuids + "PtrX.npy")


        rr = []
        rs = []               
        if Y is not None:
            os.remove('/dev/shm/'+ uuids + "DataY.npy")
            for r in res:
                rr.append(r[0])
                rs += r[1]
            res = rr    
        if Y is not None:
            if avg == 'macro':
                return multiply(asarray(res).sum(axis=0), 1. / self.n_estimators),rs
            else:
                return asarray(res), rs            
        else:        
            if avg == 'macro':
                return multiply(asarray(res).sum(axis=0), 1. / self.n_estimators)
            else:
                return asarray(res)

        
    def __init__(self,C, kernel = 'linear', max_deth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000, cluster_cfg = 'servers.yml',
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,gamma=1000.,
                 intercept_scaling=1.,dropout_low=0.,dropout_high=1.0,noise=0.,cov_dr=0., criteria='gini', db_name='work_queue',\
                 db_host='localhost', res_name = 'res_queue', res_host = 'localhost'):
        self.criteria = criteria
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
        self.intercept_scaling = intercept_scaling 
        self.dropout_low = dropout_low
        self.dropout_high = dropout_high 
        self.noise = noise
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.cov_dr = cov_dr 
        self.balance = True
        self.res_name = "result.bin" 
        self.addr = ("localhost",5555)      
        self.cluster_cgf =  loadClusterCfg(cluster_cfg) 
        
        self.db = rediswq.RedisWQ(name=db_name, host=db_host)
        self.res = rediswq.RedisWQ(name=res_name, host=res_host)
        self.db_name = db_name
        self.db_host = db_host
        self.res_name = res_name
        self.res_host = res_host
