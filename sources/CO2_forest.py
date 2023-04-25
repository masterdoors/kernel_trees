import pickle
import os
import datetime
from scipy.optimize import linprog
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline, make_pipeline
from copy import deepcopy
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

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

from scipy.spatial.distance import jensenshannon 
from scipy.stats import entropy

from numpy import corrcoef
from numpy import cov
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

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
    
    #if forest.univariate_ratio > 0.:
    #    flag = numpy.random.choice([0,1], p=[forest.univariate_ratio,1 - forest.univariate_ratio])
    #    if flag == 0:
    #        k = 'univariate'
    #    else:
    #        k = forest.kernel    
    #else:
    k = forest.kernel
        
    tree = co2.CO2Tree(C=forest.C , kernel=k,\
    tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_deth,\
     min_samples_split = forest.min_samples_split,dual=forest.dual,\
    min_samples_leaf = forest.min_samples_leaf, seed = seed_,\
     sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
     gamma=forest.gamma,intercept_scaling=forest.intercept_scaling,dropout_low=forest.dropout_low,dropout_high=forest.dropout_high,noise=forest.noise,cov_dr=forest.cov_dr, criteria = forest.criteria,spatial_mul=forest.spatial_mul)
    tree.fit(x,Y, preprocess = False)
    return tree

def probber(uuids, shapex,train_shape,tree,stat_only,use_weight = True,withY = False):
        dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
        indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
        ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
        
        if withY:
            Y = load('/dev/shm/' + uuids + "DataY.npy",mmap_mode='r')
        else:
            Y = None            
        x = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32, copy=False)    
        
        dataX = load('/dev/shm/' + uuids + 'trainDataX.npy',mmap_mode='r')
        indX = load('/dev/shm/' + uuids + "trainIndX.npy",mmap_mode='r')
        ptrX = load('/dev/shm/' + uuids + "trainPtrX.npy",mmap_mode='r')    
        train_data = csr_matrix((dataX,indX,ptrX), shape=train_shape,dtype=numpy.float32,copy=False)
    
        return tree.predict_proba(x,Y,train_data, preprocess = False,stat_only=stat_only,use_weight=use_weight)    

def indicator(uuids, shapex,train_shape,tree,noise,balance_noise):
    dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
    indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
    ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
    
    dataX = load('/dev/shm/' + uuids + 'trainDataX.npy',mmap_mode='r')
    indX = load('/dev/shm/' + uuids + "trainIndX.npy",mmap_mode='r')
    ptrX = load('/dev/shm/' + uuids + "trainPtrX.npy",mmap_mode='r')    
    train_data = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32,copy=False)    

    x = csr_matrix((dataX,indX,ptrX), shape=train_shape,dtype=numpy.float32, copy=False)    
    return tree.getIndicators(x, train_data, noise = noise, balance_noise = balance_noise)     

def statter(tree):
    return tree.getWeights()    

def jsd(norm,i):
    sum_ = entropy(norm[i]) * norm.shape[0]
    
    for j in range(norm.shape[0]):
        if i != j:
            sum_ += entropy(norm[i],norm[j]) + entropy(norm[j],norm[i]) + entropy(norm[j]) 
    return float(sum_) / norm.shape[0]                      

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

def testerL2(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data = splits[i][0]
            lr_data_test = splits[i][1]
 
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            lr_data_test = mm.transform(lr_data_test)            

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, splits_Y[i][0])

            to_remove,before,after = tr.prune(lr_data, lr.coef_, n)
            lr_data = tr.do_prune(lr_data,to_remove)

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)                    
            lr.fit(lr_data, splits_Y[i][0])

            lr_data = tr.do_prune(lr_data_test,to_remove)
            y_pred_ = lr.predict(lr_data)                         
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))
            
    return c,n,numpy.asarray(tests).mean() + numpy.asarray(tests).std()

def testerNoise(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data_test = splits[i][1]
            lr_data = tr.addNoise(splits[i][0],n) 
            
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            lr_data_test = mm.transform(lr_data_test)               

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, splits_Y[i][0])
            y_pred_ = lr.predict(lr_data_test)                 
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))

    return c,n,numpy.asarray(tests).mean() + numpy.asarray(tests).std()             

def testerNaive(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data = splits[i][0]
            lr_data_test = splits[i][1]

            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            
            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)                    


            lr.fit(lr_data, splits_Y[i][0])
            coeffs = numpy.abs(lr.coef_)

            max_coefs = coeffs.max(axis=1)
            remain_idxs = numpy.zeros((lr.coef_.shape[1],)).astype(bool)
            for j in range(max_coefs.shape[0]):
                remain_idxs = numpy.logical_or(remain_idxs,coeffs[j] >= max_coefs[j] * n) 

            lr_data_tr = lr_data[:,remain_idxs]  
            lr = LogisticRegression(C=c,
                        fit_intercept=False,
                        solver='lbfgs',
                        max_iter=1000,
                        multi_class='multinomial', n_jobs=-1)                    
            lr.fit(lr_data_tr, splits_Y[i][0])
            lr_data = lr_data_test
            lr_data = mm.transform(lr_data)  
            
            y_pred_ = lr.predict(lr_data[:,remain_idxs])                        
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))
   
    return c,n,numpy.asarray(tests).mean() + numpy.asarray(tests).std()           

def testerL2_noise(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n,n2):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data = splits[i][0]
            lr_data_test = splits[i][1]
            lr_data = tr.addNoise(lr_data,n2) 
      
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            lr_data_test = mm.transform(lr_data_test)               

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, splits_Y[i][0])
            
            #print("fit train",c,n,n2)

            to_remove,before,after = tr.prune(lr_data, lr.coef_, n)
            lr_data = tr.do_prune(lr_data,to_remove)

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)                    
            lr.fit(lr_data, splits_Y[i][0])
            #print("fit test",c,n,n2)
            
            lr_data = tr.do_prune(lr_data_test,to_remove)
            y_pred_ = lr.predict(lr_data)                         
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))
    return c,n,n2,numpy.asarray(tests).mean() + numpy.asarray(tests).std()

def inter_log(*args):
    print (args)
    args_ = [str(a) for a in args]
    with open("runlog.txt","a") as f:
        f.write(", ".join(args_) + "\n")

class BaseCO2Forest:
    def eliminatedForest(self):
        that = deepcopy(self)
        for i in range(that.n_estimators):
            for j in range(len(that.trees[i].nodes)):
                that.trees[i].nodes[j].model = None  
                that.trees[i].nodes[j].features_weight = None 
                that.trees[i].nodes[j].counts = None
        return that
    
    def stat(self):
        return Parallel(n_jobs=self.n_jobs,backend="loky")(delayed(statter)(t) for t in self.trees)
    
    def sequential_fit(self,x,Y):
        self.trees = []
        forest = self
        for i in range(self.n_estimators):
            tree = co2.CO2Tree(C=forest.C , kernel=forest.kernel,\
            tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_deth,\
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
    
    def reinforce_prune(self,n,C,X,Y,sample_weigths = None):
        lr = LogisticRegression(C=C,
                        fit_intercept=False,
                        solver='lbfgs',
                        max_iter=100,
                        multi_class='multinomial', n_jobs=-1)            

        lr_data = self.getIndicators(X)
        mm = make_pipeline(MinMaxScaler(), Normalizer())
        mm.fit(lr_data)

        lr_data = mm.transform(lr_data)               
        lr.fit(lr_data, Y) 

        to_remove,best_before,best_after = self.prune(lr_data, lr.coef_, n)
        lr_data = self.do_prune(lr_data,to_remove) 
        self.to_remove = to_remove
        lr = LogisticRegression(C=C,
                        fit_intercept=False,
                        solver='lbfgs',
                        max_iter=100,
                        multi_class='multinomial', n_jobs=-1) 
        lr.fit(lr_data, Y,sample_weigths) 
        self.lr = lr
                        

    #@profile
    def fit(self,x,Y,x_test=None, Y_test=None,model=False, sample_weights = None):
        
        x = csr_matrix(x)
        
        self.x = x
        
        self.le = LabelEncoder().fit(Y)
        Y = self.le.transform(Y)
        if Y_test is not None:
            Y_test = self.le.transform(Y_test)
     
        if not model:
            uuids = str(uuid.uuid4())

            save('/dev/shm/'+ uuids + "DataX",x.data)
            save('/dev/shm/'+ uuids + "IndX",x.indices)
            save('/dev/shm/'+ uuids + "PtrX",x.indptr)
            save('/dev/shm/'+ uuids + "DataY",Y) 

            self.trees = Parallel(n_jobs=self.n_jobs,backend="loky")(delayed(fitter)(uuids,self,x.shape,i+(self.id_*self.n_estimators + 1)) for i in range(self.n_estimators))

            os.remove('/dev/shm/'+ uuids + "DataX.npy")
            os.remove('/dev/shm/'+ uuids + "IndX.npy")
            os.remove('/dev/shm/'+ uuids + "PtrX.npy")
            os.remove('/dev/shm/'+ uuids + "DataY.npy")   
        else:
            with open('forest.pickle', 'rb') as f:
                self.trees = pickle.load(f).trees     
        
        if self.reinforced:
            self.reinforce_prune(self.prune_level,self.reC,x,Y,sample_weights)
            
        if self.cov_dr > 0:   
            with open('forest.pickle', 'wb') as f:
                pickle.dump(self, f)              
            #cprobs = self.sequential_predict_proba(x,use_weight=False)
            #r2 = cov(numpy.hstack([cprobs[:,:,layer] for layer in range(1,cprobs.shape[2])]))
            #r2 = numpy.sqrt(r2.sum()) 

            kf = StratifiedKFold(n_splits=3, shuffle=True)
            #kf = KFold(n_splits=3, shuffle=True)
            self.splits = []
            self.splits_Y = []
            self.main_inds_train = self.getIndicators(x)
            self.main_inds_test = self.getIndicators(x_test)
            self.forests = []
            
            inter_log("Training forests")
            for train, test in kf.split(x,Y):         
                tr = CO2Forest(C=self.C, dual=self.dual,
                                 tol = self.tol,max_iter=self.max_iter,kernel=self.kernel,
                                 max_deth=self.max_deth,n_jobs=self.n_jobs,sample_ratio=self.sample_ratio, 
                                 feature_ratio = self.feature_ratio,n_estimators=self.n_estimators,
                                 gamma=self.gamma,dropout_low=self.dropout_low,dropout_high=self.dropout_high,
                                 noise=self.noise,cov_dr=0,criteria=self.criteria,spatial_mul=self.spatial_mul)
                tr.fit(csr_matrix(x[train]),Y[train])
                #self.forests.append(tr.eliminatedForest())
                self.forests.append(tr)
                self.splits_Y.append([Y[train],Y[test]])
                self.splits.append([tr.getIndicators(x[train]),tr.getIndicators(x[test])])
                #del tr
                
                with open('forest.pickle', 'wb') as f:
                    pickle.dump(self, f)                 
                
            inter_log("Done")   
    
    def    do_tests(self,x,Y,x_test, Y_test):       
            Y = self.le.transform(Y)
            Y_test = self.le.transform(Y_test)
        
            best = 1.
            best_c = 0
            best_n = 0
            best_n2 = 0
            best_r3 = 0
            best_before = 0
            best_after = 0
            
            splits = self.splits
            splits_Y = self.splits_Y
            
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerL2_noise)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n,n2) for n2 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,n2,t in tests:
                if t < best:
                    inter_log ("New best:",t,c,n,n2)                            
                    best = t  
                    best_c = c
                    best_n = n  
                    best_n2 = n2
                     
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.addNoise(self.main_inds_train,best_n2)
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
             
            lr.fit(lr_data, Y) 
            
            to_remove,best_before,best_after = self.prune(lr_data, lr.coef_, best_n)
            lr_data = self.do_prune(lr_data,to_remove)            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1) 
            lr.fit(lr_data, Y)  
            #best_r3 = self.get_cov(lr_data, lr.coef_) 

            lr_data = mm.transform(self.main_inds_test)              
            
            lr_data = self.do_prune(lr_data,to_remove)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None) 
            best_f1m = f1_score(y_pred_,Y_test,average='macro')            
            inter_log ("LR l2-norm pruning + noise test result:", best, "f1:",best_f1,best_f1m,"C:", best_c, "pruning:", best_n, "noise:", best_n2,"instability before:", best_before, "instability after:", best_after)              
            
            best = 1.
            best_c = 0
            for c in [0.01,0.1,1.0,10]:
                tests = []
                orig_tests = []
                for i in range(len(splits)): 
                    tr = self.forests[i]

                    lr_data = splits[i][0]
                    mm = make_pipeline(MinMaxScaler(), Normalizer())
                    mm.fit(lr_data)

                    lr_data = mm.transform(lr_data)                       
                    
                    lr = LogisticRegression(C=c,
                                    fit_intercept=False,
                                    solver='lbfgs',
                                    max_iter=100,
                                    multi_class='multinomial', n_jobs=-1)

                    lr.fit(lr_data, splits_Y[i][0])
                    lr_data = splits[i][1]

                    lr_data = mm.transform(lr_data)                       

                    y_pred_ = lr.predict(lr_data)                 
                    tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))

                if numpy.asarray(tests).mean() + numpy.asarray(tests).std() < best:
                    inter_log ("New best:",numpy.asarray(tests).mean(), numpy.asarray(tests).std(), c)
                    best = numpy.asarray(tests).mean() + numpy.asarray(tests).std()
                    best_c = c
                    
            lr_data = self.main_inds_train
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, Y)
            #best_r3 = self.get_cov(lr_data, lr.coef_)            
            lr_data = self.main_inds_test

            lr_data = mm.transform(lr_data)               
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)                
            best_f1 = f1_score(y_pred_,Y_test,average=None)
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            inter_log ("LR orig test result:", best,"f1:",best_f1,best_f1m,"C:", best_c)            
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0                
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerNoise)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n) for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,t in tests:
                if t < best:
                    inter_log ("New best:",t,c,n)   
                    best = t  
                    best_c = c
                    best_n = n  
                
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)
            
            lr_data = self.addNoise(self.main_inds_train, best_n)     
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            
            lr.fit(lr_data, Y)
            #est_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.main_inds_test

            lr_data = mm.transform(lr_data)               
            
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None)
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            inter_log ("LR noised test result:", best,"f1:" ,best_f1,best_f1m,"C:", best_c, "noise:", best_n)
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerNaive)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n) for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,t in tests:
                if t < best:
                    inter_log ("New best:",t,c,n)   
                    best = t  
                    best_c = c
                    best_n = n  
                
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.main_inds_train
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            
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
            #est_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.main_inds_test


            lr_data = mm.transform(lr_data)               
            y_pred_ = lr.predict(lr_data[:,remain_idxs])    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None) 
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            inter_log ("LR naive pruning test result:", best,"f1:",best_f1,best_f1m,"C:", best_c, "pruning:", best_n)  
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            best_before = 0
            best_after = 0
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerL2)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n) for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,t in tests:
                if t < best:
                    inter_log ("New best:",t,c,n)                            
                    best = t  
                    best_c = c
                    best_n = n  

            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.main_inds_train
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            lr.fit(lr_data, Y) 
            
            to_remove,best_before,best_after = self.prune(lr_data, lr.coef_, best_n)
            lr_data = self.do_prune(lr_data,to_remove)            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1) 
            lr.fit(lr_data, Y)  
            #best_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.main_inds_test
            lr_data = mm.transform(lr_data)               
            lr_data = self.do_prune(lr_data,to_remove)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None) 
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            inter_log ("LR l2-norm pruning test result:", best,"f1:",best_f1,best_f1m,"C:", best_c, "pruning:", best_n,"instability before:", best_before, "instability after:", best_after)              
            
            self.trees = Parallel(n_jobs=5,backend="multiprocessing")(delayed( weighter)(i,self,numpy.ones(self.n_estimators,)) for i in range(self.n_estimators))
            y_pred_ = self.predict(csr_matrix(x_test))
            rs_local = accuracy_score(Y_test,y_pred_)
            y_pred_ = self.predict(x)
            rs_train = accuracy_score(Y,y_pred_)
            inter_log ("Train error orig:", 1. - rs_train)    
            inter_log ("Test error orig:", 1. - rs_local)    
            inter_log ("Delta orig:", rs_train - rs_local)

    def sequential_predict(self,x,use_weight=True):
        probas = []
        for tree in self.trees:
            probas.append(tree.predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  (asarray(probas).sum(axis=0), 1. / self.n_estimators)
        res =  argmax(proba, axis = 1)
        zr = res == 0
        res[zr] = 1
        return self.le.inverse_transform(res)   
    
    def sequential_predict_proba(self,x,use_weight=True):
        probas = []
        for i in range(len(self.trees)):
            probas.append(self.trees[i].predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  asarray(probas)#[:,1:]
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
        
        res = self.le.inverse_transform(res)
        if Y is not None:
            return res,cmp
        else:
            return res    
        
    def getIndicators(self,x, noise= 0., balance_noise = False):
        uuids = str(uuid.uuid4())
        
        save('/dev/shm/'+ uuids + "DataX",x.data)
        save('/dev/shm/'+ uuids + "IndX",x.indices)
        save('/dev/shm/'+ uuids + "PtrX",x.indptr)      
        save('/dev/shm/'+ uuids + "trainDataX",self.train_data.data)
        save('/dev/shm/'+ uuids + "trainIndX",self.train_data.indices)
        save('/dev/shm/'+ uuids + "trainPtrX",self.train_data.indptr)            
        

        res = Parallel(n_jobs=self.n_jobs,backend="multiprocessing")(delayed(indicator)(uuids,x.shape,train_data.shape,t,noise,balance_noise) for t in self.trees)

        os.remove('/dev/shm/'+ uuids + "DataX.npy")
        os.remove('/dev/shm/'+ uuids + "IndX.npy")
        os.remove('/dev/shm/'+ uuids + "PtrX.npy")   
        os.remove('/dev/shm/'+ uuids + "trainDataX.npy")
        os.remove('/dev/shm/'+ uuids + "trainIndX.npy")
        os.remove('/dev/shm/'+ uuids + "trainPtrX.npy")
        
        
        res = sorted(res, key=lambda tup: tup[1])
        indicators = [r for r,i in res]
           
        return numpy.hstack(indicators)    
    
    def addNoise(self,indicators, noise= 0.):
        indicators = deepcopy(indicators)
        for i in range(indicators.shape[1]):
            #print (indicators[:,i])
            nonzero = numpy.where(indicators[:,i] > 0)[0]
            idxs = numpy.random.randint(0, nonzero.shape[0], int(nonzero.shape[0]*noise)) 
            indicators[nonzero[idxs],i] = 0. #let's make the classifiers different again
        return indicators      
    
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
        x = csr_matrix(x)           
        if self.reinforced:
            inds = self.getIndicators(x)
            inds = self.do_prune(inds,self.to_remove)
            r = self.lr.decision_function(inds)
            return r
        else:    
            uuids = str(uuid.uuid4())

            save('/dev/shm/'+ uuids + "DataX",x.data)
            save('/dev/shm/'+ uuids + "IndX",x.indices)
            save('/dev/shm/'+ uuids + "PtrX",x.indptr)     
            
            save('/dev/shm/'+ uuids + "trainDataX",self.train_data.data)
            save('/dev/shm/'+ uuids + "trainIndX",self.train_data.indices)
            save('/dev/shm/'+ uuids + "trainPtrX",self.train_data.indptr)                 

            if Y is not None:
                Y = self.le.transform(Y)
                save('/dev/shm/'+ uuids + "DataY",Y) 
            res = Parallel(n_jobs=self.n_jobs,backend="loky")(delayed(probber)(uuids,x.shape,self.train_data.shape,t,False,use_weight,Y is not None) for t in self.trees)

            os.remove('/dev/shm/'+ uuids + "DataX.npy")
            os.remove('/dev/shm/'+ uuids + "IndX.npy")
            os.remove('/dev/shm/'+ uuids + "PtrX.npy")
            os.remove('/dev/shm/'+ uuids + "trainDataX.npy")
            os.remove('/dev/shm/'+ uuids + "trainIndX.npy")
            os.remove('/dev/shm/'+ uuids + "trainPtrX.npy")

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
                    return asarray(res).sum(axis=0)#[:,1:]
                else:
                    return asarray(res)#[:,1:]

        
    def __init__(self,C, kernel = 'linear', max_deth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,intercept_scaling=1.,dropout_low=0.,dropout_high=1.0,noise=0.,cov_dr=0.,\
                 criteria='gini',spatial_mul=1.0,reinforced = False, prune_level = 0.,reC=10,id_=0,univariate_ratio=0.0):
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
        #os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.cov_dr = cov_dr 
        self.spatial_mul = spatial_mul
        self.prune_level = 0
        self.reC = 10.
        self.reinforced = reinforced
        self.id_ = id_
        self.univariate_ratio = univariate_ratio


class CO2ForestClassifier(BaseCO2Forest, ClassifierMixin):
    pass

class CO2ForestRegressor(BaseCO2Forest, RegressorMixin):
    pass

