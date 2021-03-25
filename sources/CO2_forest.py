# coding: utf-8

import os
import datetime
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

from scipy.special import kl_div

from numpy import load
from numpy import save
import uuid
import numpy

from scipy.spatial.distance import jensenshannon 
from scipy.stats import entropy

#import cProfile


#def profile(func):
#    """Decorator for run function profile"""
#    def wrapper(*args, **kwargs):
#        profile_filename = func.__name__ + '.prof'
#        profiler = cProfile.Profile()
#        result = profiler.runcall(func, *args, **kwargs)
#        profiler.dump_stats(profile_filename)
#        return result
#    return wrapper


def fitter(uuids,forest,shapex,seed_):
    dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
    indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
    ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
    x = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32,copy=False)
    Y = load('/dev/shm/' + uuids + "DataY.npy",mmap_mode='r')
        
    tree = co2.CO2Tree(C=forest.C , kernel=forest.kernel,\
    tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_deth,\
     min_samples_split = forest.min_samples_split,dual=forest.dual,\
    min_samples_leaf = forest.min_samples_leaf, seed = None,\
     sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
     gamma=forest.gamma,intercept_scaling=forest.intercept_scaling,dropout_low=forest.dropout_low,dropout_high=forest.dropout_high,noise=forest.noise,cov_dr=forest.cov_dr, criteria = forest.criteria)
    tree.fit(x,Y, preprocess = False)
    return tree

       
def probber(uuids, shapex,tree,stat_only,use_weight = True,withY = False):
        dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
        indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
        ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
        
        if withY:
            Y = load('/dev/shm/' + uuids + "DataY.npy",mmap_mode='r')
        else:
            Y = None            
        x = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32, copy=False)    
        return tree.predict_proba(x,Y,preprocess = False,stat_only=stat_only,use_weight=use_weight)    

def statter(tree):
    return tree.getWeights()    

def jsd(norm,i):
    sum_ = entropy(norm[i]) * norm.shape[0]
    
    for j in range(norm.shape[0]):
        if i != j:
            sum_ += entropy(norm[i],norm[j]) + entropy(norm[j],norm[i]) + entropy(norm[j]) 
    return float(sum_) / norm.shape[0]                      

def weighter(tree,forest,norm,diff,min_nom,uniform):
    t = forest.trees[tree]
    t.estimateChunkWeights(norm,diff,min_nom,uniform)
    return t


class CO2_forest:
    def stat(self):
        return Parallel(n_jobs=self.n_jobs)(delayed(statter)(t) for t in self.trees)

    #@profile
    def fit(self,x,Y):
        uuids = str(uuid.uuid4())
        
        save('/dev/shm/'+ uuids + "DataX",x.data)
        save('/dev/shm/'+ uuids + "IndX",x.indices)
        save('/dev/shm/'+ uuids + "PtrX",x.indptr)
        save('/dev/shm/'+ uuids + "DataY",Y) 
        
        #pool = Pool(self.n_jobs)
        self.trees = Parallel(n_jobs=self.n_jobs,backend="multiprocessing")(delayed(fitter)(uuids,self,x.shape,i) for i in range(self.n_estimators))
            
        #self.trees = pool.map(func,range(self.n_estimators))            
        #pool.close()
        #pool.join()
        
        os.remove('/dev/shm/'+ uuids + "DataX.npy")
        os.remove('/dev/shm/'+ uuids + "IndX.npy")
        os.remove('/dev/shm/'+ uuids + "PtrX.npy")
        os.remove('/dev/shm/'+ uuids + "DataY.npy")   
        
        if self.cov_dr > 0:               
        
            counts = []  
            for t in self.trees:
                counts.append(t.getCounts())
    
            norm = numpy.concatenate(counts)
            #norm = asarray(norm,float) / (norm.sum(axis=1).reshape(-1,1)) + 0.0001 
            
            self.covs = numpy.zeros((norm.shape[1],norm.shape[1]))
            
            norm = norm.sum(axis=1)
                 
            #means = norm.mean(axis=1) 
            #for i in range(norm.shape[1]):
                #nm = norm[:,i].mean()
            #    for j in range(norm.shape[1]):
                    #self.covs[i,j] =  ((norm[:,i] - nm)*(norm[:,j] - norm[:,j].mean())).mean()
            #        cvs = (norm[:,i] - means[i])*(norm[:,j] - means[j])
                    #cvs[cvs < 0.] = 0.           
            #        self.covs[i,j] =  cvs.mean()  
                    
            #print  (datetime.datetime.now())
            #chain_covs = Parallel(n_jobs=self.n_jobs)(delayed(jsd)(norm,i) for i in range(norm.shape[0]))
            #div_arr = 0.5*kl_div(norm,avg)
            #sm = div_arr[i].sum()*norm.shape[0] + div_arr[:i].sum() + div_arr[i+1:].sum()
                 
            #print  (datetime.datetime.now())
            #chain_covs = numpy.asarray(chain_covs)
            #chain_covs = norm.dot(numpy.transpose(1. - norm)).sum(axis=1)

            min_cov =  norm.min()       
            max_cov =  norm.max()
            #avg_cov = (max_cov + min_cov) / 2.0
            max_diff = max_cov - min_cov

            if not self.cov_dr == 0.:
                global_counter = 0
                for t in self.trees:
                    for lidx in t.leaves:
                        dst = float(norm[global_counter] - min_cov) / max_diff
                        #dst = 1. / (numpy.exp(-2* dst + 1) + 1)
                        #dst = chain_covs[global_counter]
                        t.nodes[lidx].chunk_weight = dst 
                        global_counter += 1
            #self.trees = Parallel(n_jobs=self.n_jobs,backend="multiprocessing")(delayed( weighter)(i,self,norm, diff,min_nom,uniform)for i in range(self.n_estimators))


    def predict(self,x,Y=None,use_weight=True):
        if Y is not None:
            proba, cmp = self.predict_proba(x,Y,use_weight=use_weight)
        else:
            proba = self.predict_proba(x,Y,use_weight=use_weight)    
        res =  argmax(proba, axis = 1)
        zr = res == 0
        res[zr] = 1
        if Y is not None:
            return res,cmp
        else:
            return res         
    
    def predict_proba(self,x,Y=None,avg='macro',use_weight=True):
        uuids = str(uuid.uuid4())
        
        save('/dev/shm/'+ uuids + "DataX",x.data)
        save('/dev/shm/'+ uuids + "IndX",x.indices)
        save('/dev/shm/'+ uuids + "PtrX",x.indptr)        

        if Y is not None:
            save('/dev/shm/'+ uuids + "DataY",Y) 
        res = Parallel(n_jobs=self.n_jobs)(delayed(probber)(uuids,x.shape,t,False,use_weight,Y is not None) for t in self.trees)

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
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,gamma=1000.,intercept_scaling=1.,dropout_low=0.,dropout_high=1.0,noise=0.,cov_dr=0., criteria='gini'):
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

        
        
