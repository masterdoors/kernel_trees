# kernel_trees
Bagging on kernel trees. See: *Devyatkin, D. A., & Grigoriev, O. G. (2022). Random Kernel Forests. IEEE Access, 10, 77962-77979.*.

## Table of Contents:
- [Dependences](#dependences)
- [Installation](#installation)
- [API](#api)
- [Examples](#examples)
    - [Basic usage](#basic-usage)
    - [Refined Forest](#refined-forest)
    - [Cascade Forest Regressors](#cascade-forest-regressors)

## Code structure:

sources - source files: 

--sources/cascade - adopted Deep Forest

--sources/distributed - experimental distributed implementation. 

--sources/regularization - experiments with ensemble pruning.

--sources/CO2_**.py - Kernel Forest.

tests - files and notebooks to run various experiments.

<a name="dependences"/>

# Dependences:
(One can use pip to install all of them)
1. Python 3.8.16

2. Cython

3. numpy/scipy 1.21.0/1.5.0 or higher

4. sympy 1.7 or higher

5. joblib 1.0.0 or higher

6. Fork of the scikit-learn 0.22 (use pip install git+https://github.com/masterdoors/scikit-learn to install). 

<a name="installation"/>

# Installation:
## Kernel Forests
pip install git+https://github.com/masterdoors/kernel_trees.git
## Kernel Forests with GPU acceleration
1. Install the fork of ThunderSVM (It supports sample_weights): https://github.com/masterdoors/thundersvm
2. Install Kernel Forests: pip install git+https://github.com/masterdoors/kernel_trees.git
## Cascade Forests
1. Go to sources/cascade. Run setup.py install (Note that the Cascade Forest has its own licence, read it beforehead)
2. Install Kernel Forests: pip install git+https://github.com/masterdoors/kernel_trees.git

<a name="api"/>

# API
## Kernel Forest classifier and Regressor

Class __CO2_forest.CO2ForestClassifier__ - Kernel Forest classifier.

Methods:

__\__init____(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='gini',spatial_mul=1.0,id_=0,univariate_ratio=0.0,verbose=0)*

*C* - Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive and pretty large comparing to SVM (1000 and greather).

*kernel* - {linear, polynomial, gaussian, univariate}. Specifies the kernel type to train decision stamps. The 'univariate' means the standard axis-parallel splitter.

*max_depth* - The maximum depth of the tree.

*tol* - Tolerance for stopping criterion.

*min_samples_split* - The minimum number of samples required to split an internal node.

*dual* - Select the algorithm to either solve the dual or primal optimization problem (for the 'linear' kernel only). 

*max_iter* - Hard limit on iterations within solver, or -1 for no limit.

*min_samples_leaf* - The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. 

*n_jobs* - The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. 

*n_estimators* - The number of trees in the forest. 

*sample_ratio* - The number of training samples to consider when looking for the best split. sample_ratio is a fraction *[0..1]*. 

*feature_ratio* - The number of features to consider when looking for the best split. feature_ratio is a fraction *[0..1]*.

*gamma* - Kernel coefficient for gaussian kernel. Must be non-negative.

*criteria* - {gini, gain}. The function to measure the quality of a split. 

*spatial_mul* - deprecated.

*univariate_ratio*.  - Fraction of univariate trees in the ensemble *[0..1]*.

*id_* - deprecated.

*verbose* - Enable verbose output. 

__fit(self,X,y)__

Build a forest of trees from the training set (X, y).

*X* - The training input samples (dense or sparse matrix). 

*y* - The target values

__predict(X)__

Predict class for X.

*X* - The input samples. 

__predict_proba(self,X,avg='macro')__

Predict class probabilities for *X*.

*X* - The input samples.

*avg* - {macro, None}. Defines if the method should return mean probability.

Class __CO2_forest.CO2ForestRegressor__ - Kernel Forest regressor

__\__init____(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='mse',spatial_mul=1.0, id_=0,univariate_ratio=0.0, verbose=0)*

*C* - Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive and pretty large comparing to SVM (1000 and greather).

*kernel* - {linear, polynomial, gaussian, univariate}. Specifies the kernel type to train decision stamps. The 'univariate' means the standard axis-parallel splitter.

*max_depth* - The maximum depth of the tree.

*tol* - Tolerance for stopping criterion.

*min_samples_split* - The minimum number of samples required to split an internal node.

*dual* - Select the algorithm to either solve the dual or primal optimization problem (for the 'linear' kernel only). 

*max_iter* - Hard limit on iterations within solver, or -1 for no limit.

*min_samples_leaf* - The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. 

*n_jobs* - The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. 

*n_estimators* - The number of trees in the forest. 

*sample_ratio*- The number of training samples to consider when looking for the best split. sample_ratio is a fraction *[0..1]*. 

*feature_ratio* - The number of features to consider when looking for the best split. feature_ratio is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.

*gamma* - Kernel coefficient for gaussian kernel. Must be non-negative.

*criteria* - {mse}. The function to measure the quality of a split (mean squared error is the only option now). 

*spatial_mul* - deprecated.

*univariate_ratio* - Fraction of univariate trees in the ensemble *[0..1]*.

*id_* - deprecated

*verbose* - Enable verbose output. 

__fit(self,X,y)__

Build a forest of trees from the training set (X, y).

*X* - The training input samples (dense or sparse matrix). 

*y* - The target values

__predict(self,X)__

Predict class for X.

*X* - The input samples. 

## Refined Kernel Forest classifier and Regressor

Class __CO2_refined.RefinedForestClassifier__. Refined Kernel Forest classifier.

All methods and parameters are the same as for Kernel Forest, except the __\__init____ has the following additional parameters:

*prune_threshold* - fraction of leaves to be pruned after the refinement.

*pruneC* - Regularization parameter of the refinement.


Class __CO2_refined.RefinedForestRegressor__. Refined Kernel Forest regressor.

All methods and parameters are the same as for Kernel Forest, except the __\__init____ has the following additional parameters:

*prune_threshold* - fraction of leaves to be pruned after the refinement.

*pruneC* - Regularization parameter of the refinement.


## Cascade Forest

See the documentation here: <a href="https://github.com/masterdoors/kernel_trees/tree/master/sources/cascade">Cascade Forest</a>

<a name="examples"/>

# Examples
<a name="basic-usage"/>

## Basic usage:
```python
from sklearn import datasets, metrics

from scipy.sparse import csr_matrix
from sklearn import preprocessing
from numpy import asarray
from numpy.random import randint as rint

import numpy


import CO2_forest as co2f
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import optuna



digits = datasets.load_digits()


fratio = [0.05, 0.08,0.1,0.2,0.3]
tree_deth = [4,5,6,]

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

Y =  asarray(digits.target).astype('int64')

print (numpy.unique(Y,return_counts=True))

for i in range(len(Y)):
    Y[i] = Y[i] + 1

x = preprocessing.normalize(data, copy=False, axis = 0)

 
ns = rint(0,x.shape[0], size=x.shape[0])
x = x[ns]
Y = Y[ns]

x_train, x_validate, Y_train, Y_validate = train_test_split(
    x, Y, test_size=0.5, shuffle=False
)

x_sp_t = csr_matrix(x_train,dtype=numpy.float32)#[:6000]
x_sp_v = csr_matrix(x_validate,dtype=numpy.float32)#[:3000]

def objective(trial):
    C = trial.suggest_float('C', 1000, 5500)
    d = trial.suggest_int('d', 4, 7)
    f = trial.suggest_float('f', 0.05, 0.5)
    g = trial.suggest_float('g', 0.001, 100)
    
    score = [] 
    
    kf = KFold(n_splits=3)
    for _, (train_index, test_index) in enumerate(kf.split(x_sp_t)):
        trc = co2f.CO2ForestClassifier(C=C, dual=False,tol = 0.0000001,max_iter=1000000,kernel='gaussian',\
                                   max_depth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = f,\
                                   n_estimators=30,\
                                   gamma=g,criteria='gain')        

        trc.fit(x_sp_t[train_index], Y_train[train_index])
        Y_v = trc.predict(x_sp_t[test_index])
        score.append(accuracy_score(Y_train[test_index],Y_v))
                    
    return numpy.asarray(score).mean()#

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


C = study.best_trial.params["C"]
g = study.best_trial.params["g"]
d = study.best_trial.params["d"]
f = study.best_trial.params["f"]

trc = co2f.CO2ForestClassifier(C=C, dual=False,tol = 0.0000001,max_iter=1000000,kernel='gaussian',\
                           max_depth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = f,\
                           n_estimators=30,\
                           gamma=g,criteria='gain')        

trc.fit(x_sp_t, Y_train)
Y_v = trc.predict(x_sp_v)

print(
    f"Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)
```
<a name="refined-forest"/>

## Refined Forest:
Refined Kernel forest based on the method from the paper:

*Ren, S., Cao, X., Wei, Y., & Sun, J. (2015). Global refinement of random forest. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 723-730).*

```python
from sklearn import datasets, metrics

from sklearn import preprocessing
from numpy import asarray
from numpy.random import randint as rint

import numpy


import CO2_refined as co2f
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import optuna



digits = datasets.load_digits()


fratio = [0.05, 0.08,0.1,0.2,0.3]
tree_deth = [4,5,6,]

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

Y =  asarray(digits.target).astype('int64')

print (numpy.unique(Y,return_counts=True))

for i in range(len(Y)):
    Y[i] = Y[i] + 1

x = preprocessing.normalize(data, copy=False, axis = 0)

 
ns = rint(0,x.shape[0], size=x.shape[0])
x = x[ns]
Y = Y[ns]

x_train, x_validate, Y_train, Y_validate = train_test_split(
    x, Y, test_size=0.5, shuffle=False
)

x_sp_t = x_train
x_sp_v = x_validate

def objective(trial):
    C = trial.suggest_float('C', 1000, 5500)
    d = trial.suggest_int('d', 4, 8)
    f = trial.suggest_float('f', 0.01, 0.5)
    g = trial.suggest_float('g', 0.001, 100)
    thrx = trial.suggest_float('thrx', 0.01, 0.99)
    nc = trial.suggest_float('nc', 0.01, 10)
    
    score = [] 
    
    kf = KFold(n_splits=3)
    for _, (train_index, test_index) in enumerate(kf.split(x_sp_t)):
        trc = co2f.RefinedForestClassifier(C=C, dual=False,tol = 0.0000001,max_iter=1000000,kernel='gaussian',\
                                   max_depth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = f,\
                                   n_estimators=30,\
                                   gamma=g,criteria='gain', prune_threshold=thrx, pruneC=nc)        

        trc.fit(x_sp_t[train_index], Y_train[train_index])
        Y_v = trc.predict(x_sp_t[test_index])
        score.append(accuracy_score(Y_train[test_index],Y_v))
                    
    return numpy.asarray(score).mean()#

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


C = study.best_trial.params["C"]
g = study.best_trial.params["g"]
d = study.best_trial.params["d"]
f = study.best_trial.params["f"]
thrx = study.best_trial('thrx')
nc = study.best_trial('nc')

trc = co2f.RefinedForestClassifier(C=C, dual=False,tol = 0.0000001,max_iter=1000000,kernel='gaussian',\
                           max_depth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = f,\
                           n_estimators=30,\
                           gamma=g,criteria='gain', prune_threshold=thrx, pruneC=nc)        

trc.fit(x_sp_t, Y_train)
Y_v = trc.predict(x_sp_v)

print(
    f"Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)
```
<a name="cascade-forest-regressors"/>

## Cascade Forest Regressors:
Cascade Forest is presented in the paper:

*Zhou, Z. H., & Feng, J. (2017, August). Deep Forest: Towards An Alternative to Deep Neural Networks. In IJCAI (pp. 3553-3559).*

We use a modification of the implementation from [Deep Forest](https://github.com/LAMDA-NJU/Deep-Forest). 

A combination of the Cascade forest, Kernel forest, and Forest Refinement is tested in:

*Devyatkin, D. A. (2023). Estimation of vegetation indices with Random Kernel Forests. IEEE Access, 11, 29500-29509.*

The code below is a toy-dataset example of Deep Forest Cascade Regression with different basic ensembles: Random Forests, Kernel Forests, and Refined Kernel Forest (The original toy dataset generation code is borrowed from Scikit-Learn).

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import CO2_forest as co2f
import CO2_refined as co2f_re
from deepforest import CascadeForestRegressor
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(400, 1), axis=0)
y = np.sin(X).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(80))

est = [RandomForestRegressor(max_depth=4) for _ in range(2)]

model = CascadeForestRegressor(max_layers=3)
model.set_estimator(est)  


lw = 2

kernel_label = ["RF", "Cascade KRF","Reg Cascade KRF"]
model_color = ["m", "c", "g"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)

model.fit(X, y)
axes[0].plot(
    X,
    model.predict(X),
    color=model_color[0],
    lw=lw,
    label="{} model".format(kernel_label[0]),
)

axes[0].scatter(
    X,
    y,
    facecolor="none",
    edgecolor="k",
    s=50,
    label="other training data",
)


est = [co2f.CO2ForestRegressor(C=3000, dual=False,tol = 0.001,max_iter=1000000,kernel='linear',\
                                   max_depth=3,n_jobs=10,feature_ratio = 0.5,\
                                   n_estimators=100) for i in range(int(2))]
                                
model = CascadeForestRegressor(max_layers=3)
model.set_estimator(est)  


model.fit(X, y)
axes[1].plot(
    X,
    model.predict(X),
    color=model_color[0],
    lw=lw,
    label="{} model".format(kernel_label[0]),
)

axes[1].scatter(
    X,
    y,
    facecolor="none",
    edgecolor="k",
    s=50,
    label="other training data",
)

est = [co2f_re.RefinedForestRegressor(C=3000, dual=False,tol = 0.001,max_iter=1000000,kernel='linear',\
                                   max_depth=3,n_jobs=10,feature_ratio = 0.5,\
                                   n_estimators=100, prune_threshold=0.1, pruneC=10000.0) for i in range(int(2))]
                                
model = CascadeForestRegressor(max_layers=3)
model.set_estimator(est)  


model.fit(X, y)
axes[2].plot(
    X,
    model.predict(X),
    color=model_color[0],
    lw=lw,
    label="{} model".format(kernel_label[0]),
)

axes[2].scatter(
    X,
    y,
    facecolor="none",
    edgecolor="k",
    s=50,
    label="other training data",
)


fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Cascade (deep) Forest: RFR vs KFR vs refined KFR", fontsize=14)
plt.show()
```

The results are presented below:

![Results of the Cascade Regression](https://github.com/masterdoors/kernel_trees/blob/master/train_reg_res.png)
