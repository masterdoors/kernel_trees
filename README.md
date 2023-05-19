# kernel_trees
Bagging on kernel trees

Now it uses LinearSGD solver to train oblique splits. Please build and install the patched Scikit-learn first (from the thirdparty directory) if you want to use LinearSVC instead.

sources - source files: 

--sources/cascade - adopted Deep Forest

--sources/distributed - experimental distributed implementation. 

--sources/regularization - experiments with ensemble pruning.

--sources/CO2_**.py - Kernel Forest.

tests - files and notebooks to run various experiments.

thirdparty - files to patch Scikit-learn (deprecated).

# Dependences:
(One can use pip to install all of them)
1. Python 3.8.16

2. Cython

3. numpy/scipy 1.21.0/1.5.0 or higher

4. sympy 1.7 or higher

5. joblib 1.0.0 or higher

6. Fork of the scikit-learn 0.22 (use pip install git+https://github.com/masterdoors/scikit-learn to install). 

# Installation:
## Kernel Forests
pip install git+https://github.com/masterdoors/kernel_trees.git
## Kernel Forests with GPU acceleration
1. Install the fork of ThunderSVM (It supports sample_weights): https://github.com/masterdoors/thundersvm
2. Install Kernel Forests: pip install git+https://github.com/masterdoors/kernel_trees.git
## Cascade Forests
1. Go to sources/cascade. Run setup.py install
2. Install Kernel Forests: pip install git+https://github.com/masterdoors/kernel_trees.git

# Basic usage:
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

# Refined Forest:
Refined Kernel forest based on the method from the paper:

*Ren, S., Cao, X., Wei, Y., & Sun, J. (2015). Global refinement of random forest. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 723-730).*

```
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
