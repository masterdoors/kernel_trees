# coding: utf-8
import nlopt
# !pip install nlopt

import os
import datetime
from scipy.optimize import linprog
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#os.environ["OPENBLAS_NUM_THREADS"] = "1"

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

# +
from scipy.spatial.distance import jensenshannon 
from scipy.stats import entropy

from numpy import corrcoef
from numpy import cov
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
# -

# import cProfile


# def profile(func):
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
    #if seed_ % 2 == 0:
    #    k = 'polynomial'
    #else:
    #    k = 'gaussian'
        
    tree = co2.CO2Tree(C=forest.C , kernel=forest.kernel,\
    tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_depth,\
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

# +
def weighter(tree,forest,w, offset = 0):
    #todo: offset
    t = forest.trees[tree]
    if len(w.shape) == 1:
        t.estimateChunkWeights(w[tree])
    else:
        w_ = numpy.zeros((t.leaves_number,w.shape[0]+1))
        for i in range(1,w.shape[0]+1):
            w_[:,i] = w[i - 1,offset:offset + t.leaves_number]
        t.estimateChunkWeights(w_)        
    return t

#def get_cov(counts):
#    res = numpy.zeros((counts.shape[0],counts.shape[0]))
#    for i in range(counts.shape[0]):
#        for j in range(counts.shape[0]):
#            sm = 0
#            total = 0
#            for k in range(counts.shape[1]):
#                if counts[i,k] == 1 and counts[j,k] == 1:
#                    sm += 1
#                if counts[i,k] == 1 or counts[j,k] == 1:
#                    total += 1    
#            res[i,j] = float(sm) / total
#    return res        


# -

class CO2Forest:
    def stat(self):
        return Parallel(n_jobs=self.n_jobs)(delayed(statter)(t) for t in self.trees)
    
    def sequential_fit(self,x,Y):
        self.trees = []
        forest = self
        for i in range(self.n_estimators):
            tree = co2.CO2Tree(C=forest.C , kernel=forest.kernel,\
            tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_depth,\
            min_samples_split = forest.min_samples_split,dual=forest.dual,\
            min_samples_leaf = forest.min_samples_leaf, seed = None,\
            sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
            gamma=forest.gamma,intercept_scaling=forest.intercept_scaling,dropout_low=forest.dropout_low,dropout_high=forest.dropout_high,noise=forest.noise,cov_dr=forest.cov_dr, criteria = forest.criteria)

            tree.fit(x,Y, preprocess = False)
            self.trees.append(tree)
            
    def prune(self, indicators, coefs, ratio):
        offset = 0
        j = 0
        norm_sums = {}
        for i in range(len(self.trees)):
            while j - offset < self.trees[i].leaves_number:
                if j - offset in self.trees[i].adj_leaves:
                    v1 = numpy.asarray([coefs[k,j] for k in range(coefs.shape[0])])
                    v2 = numpy.asarray([coefs[k,j + 1] for k in range(coefs.shape[0])])
                    norm_sums[j] = numpy.linalg.norm(v1) + numpy.linalg.norm(v2)
                j += 1
            offset += self.trees[i].leaves_number        
        to_remove = dict(sorted(norm_sums.items(), key=lambda kv: kv[1])[:int(len(norm_sums)*ratio)])
        j = 0
        offset = 0
        pr_before = 0
        pr_after = 0
        for i in range(len(self.trees)):
            while j - offset < self.trees[i].leaves_number:
                if (j - offset in self.trees[i].adj_leaves) and (j in to_remove):
                    pr_before += sum(self.trees[i].adj_leaves[j - offset])
                    pr_after += self.trees[i].adj_leaves[j - offset][0]
                j += 1
            offset += self.trees[i].leaves_number          
 
        return to_remove, pr_before / len(self.trees),pr_after / len(self.trees) 
    
    def do_prune(self, indicators, to_remove):
        for i in to_remove:
            indicators[:,i] = indicators[:,i] + indicators[:,i +1]
        indicators = numpy.delete(indicators, [idx + 1 for idx in to_remove], axis=1)
        indicators[indicators > 1.] = 1.   
        return indicators  
    
    def get_cov(self, X, coef):
        coef = coef - coef.min(axis=0)
        coef = coef / (coef.sum(axis=0) + 0.000000001)
        probas = numpy.zeros((X.shape[1],X.shape[0],coef.shape[0]))
        for i in range(probas.shape[0]):
            for j in range(probas.shape[1]):
                for k in range(probas.shape[2]):
                    probas[i,j,k] = X[j,i] * coef[k,i]
                    
        cov_ = cov(numpy.hstack([probas[:,:,layer] for layer in range(1,probas.shape[2])]))                    
        return numpy.sqrt(cov_.sum())
        
    #@profile
    def fit(self,x,Y,x_test=None, Y_test=None):
        uuids = str(uuid.uuid4())
        
        save('/dev/shm/'+ uuids + "DataX",x.data)
        save('/dev/shm/'+ uuids + "IndX",x.indices)
        save('/dev/shm/'+ uuids + "PtrX",x.indptr)
        save('/dev/shm/'+ uuids + "DataY",Y) 
        
        self.trees = Parallel(n_jobs=self.n_jobs,backend="multiprocessing")(delayed(fitter)(uuids,self,x.shape,i) for i in range(self.n_estimators))

        os.remove('/dev/shm/'+ uuids + "DataX.npy")
        os.remove('/dev/shm/'+ uuids + "IndX.npy")
        os.remove('/dev/shm/'+ uuids + "PtrX.npy")
        os.remove('/dev/shm/'+ uuids + "DataY.npy")   
        
        if self.cov_dr > 0:               
            cprobs = self.sequential_predict_proba(x,use_weight=False)
            r2 = cov(numpy.hstack(cprobs[:,:,layer] for layer in range(1,cprobs.shape[2])))
            r2 = numpy.sqrt(r2.sum()) 

            best = 1.
            best_c = 0
            lr_data = self.getIndicators(x,noise = 0)
            for c in [0.01,0.1,1.0,10]:
                kf = KFold(n_splits=5, shuffle=True)
                tests = []
                for train, test in kf.split(lr_data): 
                    lr = LogisticRegression(C=c,
                                    fit_intercept=False,
                                    solver='lbfgs',
                                    max_iter=100,
                                    multi_class='multinomial', n_jobs=-1)


                    lr.fit(lr_data[train], Y[train])
                    y_pred_ = lr.predict(lr_data[test])                 
                    tests.append(1. - accuracy_score(Y[test],y_pred_))
            

                if numpy.asarray(tests).mean() < best:
                    best = numpy.asarray(tests).mean()
                    best_c = c

            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, Y)
            best_r3 = self.get_cov(lr_data, lr.coef_)            
            lr_data = self.getIndicators(csr_matrix(x_test),noise=0)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)                
            print ("LR orig test result:", best,"C:", best_c,"r2 before:",r2,"r2 after:", best_r3)            
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            for c in [0.01,0.1,1.0,10]:
                for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    lr_data = self.getIndicators(x,noise = n)
                    lr_data_test = self.getIndicators(x,noise = 0)
                    kf = KFold(n_splits=5, shuffle=True)
                    tests = []
                    for train, test in kf.split(lr_data):                     
                        lr = LogisticRegression(C=c,
                                        fit_intercept=False,
                                        solver='lbfgs',
                                        max_iter=100,
                                        multi_class='multinomial', n_jobs=-1)

                        lr.fit(lr_data[train], Y[train])
                        y_pred_ = lr.predict(lr_data_test[test])                 
                        tests.append(1. - accuracy_score(Y[test],y_pred_))
            

                    if numpy.asarray(tests).mean() < best:
                        best = numpy.asarray(tests).mean()                        
                        best_c = c
                        best_n = n
            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)
            
            lr_data = self.getIndicators(x,noise = best_n)            
            lr.fit(lr_data, Y)
            best_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.getIndicators(csr_matrix(x_test),noise=0)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)

            print ("LR noised test result:", best, "C:", best_c, "noise:", best_n, "r2 before:",r2,"r2 after:", best_r3)
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            for c in [0.01,0.1,1.0,10]:
                for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    lr_data = self.getIndicators(x,noise = n, balance_noise=True)
                    lr_data_test = self.getIndicators(x,noise = 0)
                    kf = KFold(n_splits=5, shuffle=True)
                    tests = []
                    for train, test in kf.split(lr_data):                     
                        lr = LogisticRegression(C=c,
                                        fit_intercept=False,
                                        solver='lbfgs',
                                        max_iter=100,
                                        multi_class='multinomial', n_jobs=-1)

                        lr.fit(lr_data[train], Y[train])
                        y_pred_ = lr.predict(lr_data_test[test])                 
                        tests.append(1. - accuracy_score(Y[test],y_pred_))
            

                    if numpy.asarray(tests).mean() < best:
                        best = numpy.asarray(tests).mean()                        
                        best_c = c
                        best_n = n
            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)
            
            lr_data = self.getIndicators(x,noise = best_n, balance_noise=True)            
            lr.fit(lr_data, Y)
            best_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.getIndicators(csr_matrix(x_test),noise=0)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
                        
            print ("LR balanced noise test result:", best, "C:", best_c, "noise:", best_n,"r2 before:",r2,"r2 after:", best_r3)  
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            for c in [0.01,0.1,1.0,10]:
                for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    lr_data = self.getIndicators(x,noise = 0)
                    kf = KFold(n_splits=5, shuffle=True)
                    tests = []
                    for train, test in kf.split(lr_data):                     
                        lr = LogisticRegression(C=c,
                                        fit_intercept=False,
                                        solver='lbfgs',
                                        max_iter=100,
                                        multi_class='multinomial', n_jobs=-1)                    

                    
                        lr.fit(lr_data[train], Y[train])
                        coeffs = numpy.abs(lr.coef_)
 
                        max_coefs = coeffs.max(axis=1)
                        remain_idxs = numpy.zeros((lr.coef_.shape[1],)).astype(bool)
                        for i in range(max_coefs.shape[0]):
                            remain_idxs = numpy.logical_or(remain_idxs,coeffs[i] >= max_coefs[i] * n) 
                    
                        lr_data_tr = lr_data[train][:,remain_idxs]  
                        lr = LogisticRegression(C=c,
                                    fit_intercept=False,
                                    solver='lbfgs',
                                    max_iter=100,
                                    multi_class='multinomial', n_jobs=-1)                    
                        lr.fit(lr_data_tr, Y[train]) 
                        y_pred_ = lr.predict(lr_data[test][:,remain_idxs])                        
                        tests.append(1. - accuracy_score(Y[test],y_pred_))
            
                    if numpy.asarray(tests).mean() < best:
                        best = numpy.asarray(tests).mean()                        
                        best_c = c
                        best_n = n                
                
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.getIndicators(x,noise=0)
            lr.fit(lr_data, Y)            
            coeffs = numpy.abs(lr.coef_)

            max_coefs = coeffs.max(axis=1)
            remain_idxs = numpy.zeros((lr.coef_.shape[1],)).astype(bool)
            for i in range(max_coefs.shape[0]):
                remain_idxs = numpy.logical_or(remain_idxs,coeffs[i] >= max_coefs[i] * best_n) 
                    
            lr_data = lr_data[:,remain_idxs] 
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)   
            lr.fit(lr_data, Y)
            best_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.getIndicators(csr_matrix(x_test),noise=0)
            y_pred_ = lr.predict(lr_data[:,remain_idxs])    
            best = 1. - accuracy_score(Y_test,y_pred_)

            print ("LR naive pruning test result:", best, "C:", best_c, "pruning:", best_n,"r2 before:",r2,"r2 after:", best_r3)  
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            best_before = 0
            best_after = 0
            for c in [0.01,0.1,1.0,10]:
                for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    lr_data = self.getIndicators(x,noise = 0)
                    kf = KFold(n_splits=5, shuffle=True)
                    tests = []
                    for train, test in kf.split(lr_data):                     
                        lr = LogisticRegression(C=c,
                                        fit_intercept=False,
                                        solver='lbfgs',
                                        max_iter=100,
                                        multi_class='multinomial', n_jobs=-1)

                        
                        lr_data_tr = lr_data[train] 
                        lr.fit(lr_data_tr, Y[train])

                        to_remove,before,after = self.prune(lr_data_tr, lr.coef_, n)
                        lr_data_tr = self.do_prune(lr_data_tr,to_remove)

                        lr = LogisticRegression(C=c,
                                        fit_intercept=False,
                                        solver='lbfgs',
                                        max_iter=100,
                                        multi_class='multinomial', n_jobs=-1)                    
                        lr.fit(lr_data_tr, Y[train])

                        lr_data_ = self.do_prune(lr_data[test],to_remove)
                        y_pred_ = lr.predict(lr_data_)                         
                        tests.append(1. - accuracy_score(Y[test],y_pred_))
            
                    if numpy.asarray(tests).mean() < best:
                        best = numpy.asarray(tests).mean()   
                        best_c = c
                        best_n = n  
                     
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.getIndicators(x,noise=0)
            lr.fit(lr_data, Y) 
            
            to_remove,best_before,best_after = self.prune(lr_data_tr, lr.coef_, best_n)
            lr_data = self.do_prune(lr_data,to_remove)            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1) 
            lr.fit(lr_data, Y)  
            best_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.getIndicators(csr_matrix(x_test),noise=0)
            lr_data = self.do_prune(lr_data,to_remove)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)

            print ("LR l2-norm pruning test result:", best, "C:", best_c, "pruning:", best_n,"r2 before:",r2,"r2 after:", best_r3,"instability before:", best_before, "instability after:", best_after)              
            
            best = 1.
            best_c = 0
            best_n = 0
            best_n2 = 0
            best_r3 = 0
            best_before = 0
            best_after = 0
            for c in [0.01,0.1,1.0,10]:
                for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    for n2 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                        lr_data = self.getIndicators(x,noise = n2)
                        lr_data_test = self.getIndicators(x,noise = 0)
                        kf = KFold(n_splits=5, shuffle=True)
                        tests = []
                        for train, test in kf.split(lr_data):                     
                            lr = LogisticRegression(C=c,
                                            fit_intercept=False,
                                            solver='lbfgs',
                                            max_iter=100,
                                            multi_class='multinomial', n_jobs=-1)


                            lr_data_tr = lr_data[train] 
                            lr.fit(lr_data_tr, Y[train])

                            to_remove,before,after = self.prune(lr_data_tr, lr.coef_, n)
                            lr_data_tr = self.do_prune(lr_data_tr,to_remove)

                            lr = LogisticRegression(C=c,
                                            fit_intercept=False,
                                            solver='lbfgs',
                                            max_iter=100,
                                            multi_class='multinomial', n_jobs=-1)                    
                            lr.fit(lr_data_tr, Y[train])

                            lr_data_ = self.do_prune(lr_data_test[test],to_remove)
                            y_pred_ = lr.predict(lr_data_)                         
                            tests.append(1. - accuracy_score(Y[test],y_pred_))

                        if numpy.asarray(tests).mean() < best:
                            best = numpy.asarray(tests).mean()   
                            best_c = c
                            best_n = n  
                            best_n2 = n2
                     
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.getIndicators(x,noise=best_n2)
            lr.fit(lr_data, Y) 
            
            to_remove,best_before,best_after = self.prune(lr_data_tr, lr.coef_, best_n)
            lr_data = self.do_prune(lr_data,to_remove)            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1) 
            lr.fit(lr_data, Y)  
            best_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.getIndicators(csr_matrix(x_test),noise=0)
            lr_data = self.do_prune(lr_data,to_remove)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
            
            print ("LR l2-norm pruning + noise test result:", best, "C:", best_c, "pruning:", best_n, "noise:", best_n2,"r2 before:",r2,"r2 after:", best_r3,"instability before:", best_before, "instability after:", best_after)              
            
            #best = 1.
            #best_c = 0
            #best_n = 0
            #best_f = 0
            #for c in [0.01,0.1,1.0,10, 100]:
            #    for f in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            #        for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            #            lr = CO2Forest(C=c, dual=False,tol = 0.0001,max_iter=2000000000,
            #                            kernel='linear',max_depth=2,n_jobs=10,sample_ratio=1.0,
            #                            feature_ratio = f,n_estimators=10,gamma=5,dropout_low=n,dropout_high=1.0,
            #                            noise=0.,cov_dr=0,criteria='gain')

            #            lr_data = csr_matrix(self.getIndicators(x,noise = 0))
            #            lr.fit(lr_data, Y)
                   
            #            lr_data = csr_matrix(self.getIndicators(csr_matrix(x_test),noise=0))
            #            y_pred_ = lr.predict(lr_data)    
            #            test = 1. - accuracy_score(Y_test,y_pred_)
            #            if test < best:
            #                best = test
            #                best_c = c
            #                best_n = n  
            #                best_f = f
            #print ("Deep forest test result:", best, "C:", best_c, "dropout:", best_n,"f ratio:",f)              
            #w = lr.coef_  
            
            #print ("w:",w)

            #print ("Initial probs:", probs.mean(axis=0))
            #print("Weighted probs:",(w.reshape(-1,1)*probs).sum(axis=0))
            #print ("Diff:",numpy.power(probs.mean(axis=0) - (w.reshape(-1,1)*probs).sum(axis=0),2).sum())
            self.trees = Parallel(n_jobs=self.n_jobs,backend="multiprocessing")(delayed( weighter)(i,self,numpy.ones(self.n_estimators,)) for i in range(self.n_estimators))
            y_pred_ = self.predict(csr_matrix(x_test))
            rs_local = accuracy_score(Y_test,y_pred_)
            y_pred_ = self.predict(x)
            rs_train = accuracy_score(Y,y_pred_)
            print ("Train error orig:", 1. - rs_train)    
            print ("Test error orig:", 1. - rs_local)    
            print ("Delta orig:", rs_train - rs_local)
            #offsets = [0]
            #for i,t in enumerate(self.trees[:-1]):
            #    offsets.append(offsets[i - 1] + t.leaves_number)
            
            #self.trees = Parallel(n_jobs=self.n_jobs,backend="multiprocessing")(delayed( weighter)(i,self,w,offsets[i]) for i in range(self.n_estimators))
            #y_pred_ = self.predict(csr_matrix(x_test))
            #rs_local = accuracy_score(Y_test,y_pred_)
            #y_pred_ = self.predict(x)
            #rs_train = accuracy_score(Y,y_pred_)
            #print ("Train error weighted:", 1. - rs_train)    
            #print ("Test error weighted:", 1. - rs_local)    
            #print ("Delta weighted:", rs_train - rs_local)
            
            #return probs, r2,r3

    def sequential_predict(self,x,use_weight=True):
        probas = []
        for tree in self.trees:
            probas.append(tree.predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  (asarray(probas).sum(axis=0), 1. / self.n_estimators)
        res =  argmax(proba, axis = 1)
        zr = res == 0
        res[zr] = 1
        return res   
    
    def sequential_predict_proba(self,x,use_weight=True):
        probas = []
        for i in range(len(self.trees)):
            probas.append(self.trees[i].predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  asarray(probas)
        return proba      

    def predict(self,x,Y=None,use_weight=True):
        if Y is not None:
            proba, cmp = self.predict_proba(x,Y,use_weight=use_weight)
        else:
            proba = self.predict_proba(x,Y,use_weight=use_weight)    
        res =  argmax(proba, axis = 1)
        zr = res == 0
        
        #print ("Unknown errors: ", zr.astype(int).sum())
        res[zr] = 1
        if Y is not None:
            return res,cmp
        else:
            return res    
        
    def getIndicators(self,x, noise= 0., balance_noise = False):
        indicators = []
        for i in range(len(self.trees)):        
            indicators.append(self.trees[i].getIndicators(x,noise=noise, balance_noise=balance_noise))    
        return numpy.hstack(indicators)    
    
    def get_err_matrix(self,x,Y,use_weight=True):
        proba = self.predict_proba(x,Y,None,use_weight=use_weight)[0] 
        
        res =  argmax(proba, axis = 2)
        zr = res == 0
        res[zr] = 1
        
        ret_arr = []
        for r in res:
            mask = r != Y
            ret_arr.append(mask)
        return numpy.asarray(ret_arr,dtype=int)    
    
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
                return asarray(res).sum(axis=0)
            else:
                return asarray(res)

        
    def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,gamma=1000.,intercept_scaling=1.,dropout_low=0.,dropout_high=1.0,noise=0.,cov_dr=0., criteria='gini'):
        self.criteria = criteria
        self.C = C
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.kernel = kernel
        self.max_depth = max_depth
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
        #os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.cov_dr = cov_dr 

        

