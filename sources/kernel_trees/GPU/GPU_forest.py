import numpy
from kernel_trees.CO2_tree import *
from kernel_trees.CO2_forest import * 
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

import kernel_trees.GPU.GPU_decision_stamp as dst

class GPUTreeClassifier(BaseCO2Tree, ClassifierMixin):
    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_depth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = 0, \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10.,\
                 criteria='gini', spatial_mul=1.0,verbose = 0):
        super().__init__(C, tol, max_iter,kernel, dual,max_depth, \
                 min_samples_split, min_samples_leaf, seed, \
                 sample_ratio,feature_ratio,gamma, \
                 criteria, spatial_mul,verbose)
        self.decisionStampClass = dst.GPUDecisionStampClassifier

class GPUTreeRegressor(BaseCO2Tree, RegressorMixin):
    def __init__(self,C, tol, max_iter=1000,kernel = 'linear', dual = True,max_depth = None, \
                 min_samples_split = 2, min_samples_leaf = 1, seed = 0, \
                 sample_ratio=1.0,feature_ratio=1.0,gamma=10., \
                criteria='mse', spatial_mul=1.0,verbose = 0):
        super().__init__(C, tol, max_iter,kernel, dual,max_depth, \
                 min_samples_split, min_samples_leaf, seed, \
                 sample_ratio,feature_ratio,gamma, \
                 criteria, spatial_mul,verbose)
        self.decisionStampClass = dst.DecisionStampRegressor

class GPUForest:
    def fit(self,X,y):
        
        X = X.astype(dtype=numpy.float64)
        self.train_data = X
        
        self.le = LabelEncoder().fit(y)
        y = self.le.transform(y)        
        
        self.trees = []
        forest = self
        for i in range(self.n_estimators):
            tree = forest.treeClass(C=forest.C , kernel=forest.kernel,\
            tol=forest.tol, max_iter=forest.max_iter,max_depth = forest.max_depth,\
            min_samples_split = forest.min_samples_split,dual=forest.dual,\
            min_samples_leaf = forest.min_samples_leaf, seed = i,\
            sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
            gamma=forest.gamma,criteria = forest.criteria)

            tree.fit(X,y, preprocess = False)
            self.trees.append(tree)    
    
    def predict(self,X,use_weight=True):
        probas = []
        for tree in self.trees:
            probas.append(tree.predict_proba(X,None,  train_data =  self.train_data,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  numpy.asarray(probas).sum(axis=0)
        res =  numpy.argmax(proba, axis = 1)
        return self.le.inverse_transform(res)   
    
    def predict_proba(self,X,use_weight=True):
        probas = []
        for i in range(len(self.trees)):
            probas.append(self.trees[i].predict_proba(X,None,train_data =  self.train_data,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  numpy.asarray(probas)#[:,1:]
        return proba   
    
class GPUForestClassifier(GPUForest, CO2ForestClassifier):
    def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='gini',spatial_mul=1.0,id_=0,univariate_ratio=0.0,verbose=0):    
        super().__init__(C, kernel, max_depth, tol, min_samples_split , \
                 dual,max_iter,min_samples_leaf, n_jobs, n_estimators,sample_ratio,feature_ratio,\
                 gamma,criteria,spatial_mul,id_,univariate_ratio, verbose)
        self.treeClass = GPUTreeClassifier
        
     
class GPUForestRegressor(GPUForest, CO2ForestRegressor):
    def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='mse',spatial_mul=1.0, id_=0,univariate_ratio=0.0, verbose=0):
        super().__init__(C, kernel, max_depth, tol, min_samples_split , \
                 dual,max_iter,min_samples_leaf, n_jobs, n_estimators,sample_ratio,feature_ratio,\
                 gamma,criteria,spatial_mul,id_,univariate_ratio, verbose)
        self.treeClass = GPUTreeRegressor  
        
    def fit(self,X,y):
        
        X = X.astype(dtype=numpy.float64)
        self.train_data = X
        
        self.trees = []
        forest = self
        for i in range(self.n_estimators):
            tree = forest.treeClass(C=forest.C , kernel=forest.kernel,\
            tol=forest.tol, max_iter=forest.max_iter,max_depth = forest.max_depth,\
            min_samples_split = forest.min_samples_split,dual=forest.dual,\
            min_samples_leaf = forest.min_samples_leaf, seed = i,\
            sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
            gamma=forest.gamma,criteria = forest.criteria)

            tree.fit(X,y, preprocess = False)
            self.trees.append(tree)    
    
    def predict(self,X,use_weight=True):
        probas = []
        for tree in self.trees:
            probas.append(tree.predict_proba(X,None,  train_data =  self.train_data,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  numpy.asarray(probas).mean(axis=0)
        return proba           
          