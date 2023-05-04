import pickle
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from copy import deepcopy
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

#os.environ["OPENBLAS_NUM_THREADS"] = "1"

'''
Created on 27 марта 2016 г.

@author: keen
'''
import CO2_tree as co2

from numpy import argmax
from numpy import multiply
from numpy import asarray

from joblib import Parallel, delayed
from scipy.sparse.csr import csr_matrix


from numpy import load
from numpy import save
import uuid
import numpy


from scipy.stats import entropy
from numpy import cov
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def fitter(x,Y,forest,seed_):
    assert forest.treeClass is not None
    #dataX = load('/dev/shm/' + uuids + 'DataX.npy',mmap_mode='r')
    #indX = load('/dev/shm/' + uuids + "IndX.npy",mmap_mode='r')
    #ptrX = load('/dev/shm/' + uuids + "PtrX.npy",mmap_mode='r')
    #x = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32,copy=False)
    #Y = load('/dev/shm/' + uuids + "DataY.npy",mmap_mode='r')
    
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
        return Parallel(n_jobs=self.n_jobs,backend="threading")(delayed(statter)(t) for t in self.trees)
    
    def sequential_fit(self,x,Y):
        self.trees = []
        forest = self
        for i in range(self.n_estimators):
            tree = co2.BaseCO2Tree(C=forest.C , kernel=forest.kernel,\
            tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_depth,\
            min_samples_split = forest.min_samples_split,dual=forest.dual,\
            min_samples_leaf = forest.min_samples_leaf, seed = None,\
            sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
            gamma=forest.gamma,criteria = forest.criteria)

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
        x = csr_matrix(x)
        
        self.train_data = x
        
        self.le = LabelEncoder().fit(Y)
        Y = self.le.transform(Y)
        if Y_test is not None:
            Y_test = self.le.transform(Y_test)
     
        if not model:
            #uuids = str(uuid.uuid4())

            #save('/dev/shm/'+ uuids + "DataX",x.data)
            #save('/dev/shm/'+ uuids + "IndX",x.indices)
            #save('/dev/shm/'+ uuids + "PtrX",x.indptr)
            #save('/dev/shm/'+ uuids + "DataY",Y) 

            #self.trees = Parallel(n_jobs=self.n_jobs,backend="threading",require="sharedmem")(delayed(fitter)(uuids,self,x.shape,i+(self.id_*self.n_estimators + 1)) for i in range(self.n_estimators))
            self.trees = Parallel(n_jobs=self.n_jobs,backend="threading",require="sharedmem")(delayed(fitter)(x,Y,self,i+(self.id_*self.n_estimators + 1)) for i in range(self.n_estimators))            

            #os.remove('/dev/shm/'+ uuids + "DataX.npy")
            #os.remove('/dev/shm/'+ uuids + "IndX.npy")
            #os.remove('/dev/shm/'+ uuids + "PtrX.npy")
            #os.remove('/dev/shm/'+ uuids + "DataY.npy")   
        else:
            with open('forest.pickle', 'rb') as f:
                self.trees = pickle.load(f).trees     
        
        if self.reinforced:
            self.reinforce_prune(self.prune_level,self.reC,x,Y,sample_weights)
            
    def sequential_predict(self,x,use_weight=True):
        probas = []
        for tree in self.trees:
            probas.append(tree.predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  (asarray(probas).sum(axis=0), 1. / self.n_estimators)
        res =  argmax(proba, axis = 1)
        #zr = res == 0
        #res[zr] = 1
        return self.le.inverse_transform(res)   
    
    def sequential_predict_proba(self,x,use_weight=True):
        probas = []
        for i in range(len(self.trees)):
            probas.append(self.trees[i].predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  asarray(probas)#[:,1:]
        return proba      

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
        #zr = res == 0

        #print ("Unknown errors: ", zr.astype(int).sum())
        #res[zr] = 1
        
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
        

        res = Parallel(n_jobs=self.n_jobs,backend="multiprocessing",require="sharedmem")(delayed(indicator)(x,self.train_data,t,noise,balance_noise) for t in self.trees)

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

            #save('/dev/shm/'+ uuids + "DataX",x.data)
            #save('/dev/shm/'+ uuids + "IndX",x.indices)
            #save('/dev/shm/'+ uuids + "PtrX",x.indptr)     
            
            #save('/dev/shm/'+ uuids + "trainDataX",self.train_data.data)
            #save('/dev/shm/'+ uuids + "trainIndX",self.train_data.indices)
            #save('/dev/shm/'+ uuids + "trainPtrX",self.train_data.indptr)                 

            if Y is not None:
                Y = self.le.transform(Y)
                save('/dev/shm/'+ uuids + "DataY",Y) 
            res = Parallel(n_jobs=self.n_jobs,backend="threading",require="sharedmem")(delayed(probber)(x,self.train_data,t,False,use_weight,Y is not None) for t in self.trees)

            #os.remove('/dev/shm/'+ uuids + "DataX.npy")
            #os.remove('/dev/shm/'+ uuids + "IndX.npy")
            #os.remove('/dev/shm/'+ uuids + "PtrX.npy")
            #os.remove('/dev/shm/'+ uuids + "trainDataX.npy")
            #os.remove('/dev/shm/'+ uuids + "trainIndX.npy")
            #os.remove('/dev/shm/'+ uuids + "trainPtrX.npy")

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

