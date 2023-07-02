'''
Created on 16 мар. 2021 г.

@author: keen
'''
# coding: utf-8

from sklearn import datasets, metrics

from scipy.sparse import csr_matrix
from sklearn import preprocessing
from numpy import asarray
from numpy.random import randint as rint

import numpy


from kernel_trees import CO2ForestClassifier
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
    d = trial.suggest_int('d', 4, 8)
    f = trial.suggest_float('f', 0.01, 0.5)
    g = trial.suggest_float('g', 0.001, 100)
    
    score = [] 
    
    kf = KFold(n_splits=3)
    for _, (train_index, test_index) in enumerate(kf.split(x_sp_t)):
        trc = CO2ForestClassifier(C=C, dual=False,tol = 0.0000001,max_iter=1000000,kernel='gaussian',\
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

trc = CO2ForestClassifier(C=C, dual=False,tol = 0.0000001,max_iter=1000000,kernel='gaussian',\
                           max_depth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = f,\
                           n_estimators=30,\
                           gamma=g,criteria='gain')        

trc.fit(x_sp_t, Y_train)
Y_v = trc.predict(x_sp_v)

print(
    f"Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)


