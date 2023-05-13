import pickle

from copy import deepcopy
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin


'''
Created on 27 марта 2016 г.

@author: keen
'''
import CO2_tree as co2

from numpy import argmax
from numpy import asarray

from joblib import Parallel, delayed
from scipy.sparse.csr import csr_matrix


import numpy

from sklearn.preprocessing import LabelEncoder

def fitter(x,Y,forest,seed_):
    assert forest.treeClass is not None
    k = forest.kernel
        
    tree = forest.treeClass(C=forest.C , kernel=k,\
    tol=forest.tol, max_iter=forest.max_iter,max_depth = forest.max_depth,\
     min_samples_split = forest.min_samples_split,dual=forest.dual,\
    min_samples_leaf = forest.min_samples_leaf, seed = seed_,\
     sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
     gamma=forest.gamma, criteria = forest.criteria,spatial_mul=forest.spatial_mul, verbose = forest.verbose)
    tree.fit(x,Y, preprocess = False)
    return tree

def probber(x,train_data,tree,stat_only,use_weight = True,withY = False):
    return tree.predict_proba(x,None,train_data, preprocess = False,stat_only=stat_only,use_weight=use_weight)    

def indicator(x,train_data,tree,noise,balance_noise):
    return tree.getIndicators(x, train_data, noise = noise, balance_noise = balance_noise)     

def statter(tree):
    return tree.getWeights()    

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
        return Parallel(n_jobs=self.n_jobs,backend="threading")(delayed(statter)(t) for t in self.trees)
    
    def fit(self,x,Y,x_test=None, Y_test=None,model=False, sample_weights = None):
        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Fitted estimator.
        """        
        #x = csr_matrix(x)
        
        self.train_data = x
        
        self.le = LabelEncoder().fit(Y)
        Y = self.le.transform(Y)
        if Y_test is not None:
            Y_test = self.le.transform(Y_test)
     
        if not model:
            self.trees = Parallel(n_jobs=self.n_jobs,backend="threading",require="sharedmem")(delayed(fitter)(x,Y,self,i+(self.id_*self.n_estimators + 1)) for i in range(self.n_estimators))            
        else:
            with open('forest.pickle', 'rb') as f:
                self.trees = pickle.load(f).trees     
        
        if self.reinforced:
            self.reinforce_prune(self.prune_level,self.reC,x,Y,sample_weights)
            
    def predict(self,x,Y=None,use_weight=True):
        """
        Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """        
        if Y is not None:
            proba, cmp = self.predict_proba(x,Y,use_weight=use_weight)
        else:
            proba = self.predict_proba(x,Y,use_weight=use_weight)   
            
        res =  argmax(proba, axis = 1)
        res = self.le.inverse_transform(res)
        if Y is not None:
            return res,cmp
        else:
            return res    
        
    def getIndicators(self,x, noise= 0., balance_noise = False):
        
        res = Parallel(n_jobs=self.n_jobs,backend="multiprocessing",require="sharedmem")(delayed(indicator)(x,self.train_data,t,noise,balance_noise) for t in self.trees)

        res = sorted(res, key=lambda tup: tup[1])
        indicators = [r for r,_ in res]
           
        return numpy.hstack(indicators)    
    
    def predict_proba(self,x,Y=None,avg='macro',use_weight=True):
        #x = csr_matrix(x)           
        if self.reinforced:
            inds = self.getIndicators(x)
            inds = self.do_prune(inds,self.to_remove)
            r = self.lr.decision_function(inds)
            return r
        else:    
            if Y is not None:
                Y = self.le.transform(Y)

            res = Parallel(n_jobs=self.n_jobs,backend="threading",require="sharedmem")(delayed(probber)(x,self.train_data,t,False,use_weight,Y is not None) for t in self.trees)
           
            if avg == 'macro':
                return asarray(res).sum(axis=0)#[:,1:]
            else:
                return asarray(res)#[:,1:]

        
    def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='gini',spatial_mul=1.0,reinforced = False, id_=0,univariate_ratio=0.0,verbose=0):
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
        self.spatial_mul = spatial_mul
        self.prune_level = 0
        self.reC = 10.
        self.reinforced = reinforced
        self.id_ = id_
        self.univariate_ratio = univariate_ratio
        self.verbose = verbose

class CO2ForestClassifier(BaseCO2Forest, ClassifierMixin):
    """
    A random kernel  forest classifier.
    A random forest is a meta estimator that fits a number of kernel tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    Parameters
    ----------
    n_estimators : int, default=10    
    criterion : {"gini", "gain"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "gain" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
        Note: This parameter is tree-specific.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - consider `min_samples_split` as the minimum number.
    max_features : float
        The number of features to consider when looking for the best split:
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    max_samples : int or float, default=None
        the number of samples to draw from X
        to train each base estimator.
        draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.
    """    
    def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='gini',spatial_mul=1.0,reinforced = False, id_=0,univariate_ratio=0.0,verbose=0):
        super().__init__(C, kernel, max_depth, tol, min_samples_split , \
                 dual,max_iter,min_samples_leaf, n_jobs, n_estimators,sample_ratio,feature_ratio,\
                 gamma,criteria,spatial_mul,reinforced, id_,univariate_ratio,verbose)
        self.treeClass = co2.CO2TreeClassifier 

class CO2ForestRegressor(BaseCO2Forest, RegressorMixin):
    """
    A random kernel  forest regressor.
    A random forest is a meta estimator that fits a number of kernel tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    Parameters
    ----------
    n_estimators : int, default=10    
    criterion : {"mse"}, default="mse"
        The function to measure the quality of a split. 
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - consider `min_samples_split` as the minimum number.
    max_features : float
        The number of features to consider when looking for the best split:
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    max_samples : int or float, default=None
        the number of samples to draw from X
        to train each base estimator.
        draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.
    """    
        
    def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='mse',spatial_mul=1.0,reinforced = False, id_=0,univariate_ratio=0.0, verbose=0):
        super().__init__(C, kernel, max_depth, tol, min_samples_split , \
                 dual,max_iter,min_samples_leaf, n_jobs, n_estimators,sample_ratio,feature_ratio,\
                 gamma,criteria,spatial_mul,reinforced, id_,univariate_ratio, verbose)
        self.treeClass = co2.CO2TreeRegressor

