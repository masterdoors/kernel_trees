'''
Created on 16 мар. 2021 г.

@author: keen
'''
# coding: utf-8

from sklearn import metrics
from keras.datasets import cifar10

import numpy


import CO2_forest as co2f
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import optuna



(x_train, Y_train), (x_validate, Y_validate) = cifar10.load_data()


fratio = [0.05, 0.08,0.1,0.2,0.3]
tree_deth = [4,5,6,]
 
x_sp_t = x_train.reshape((x_train.shape[0],-1))
x_sp_v = x_validate.reshape((x_validate.shape[0],-1))

print(x_train)

def objective(trial):
    C = trial.suggest_float('C', 1000, 5500)
    d = trial.suggest_int('d', 4, 8)
    f = trial.suggest_float('f', 0.05, 0.5)
    g = trial.suggest_float('g', 0.001, 100)
    print(C,d,f,g)    
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
        print(score)            
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


