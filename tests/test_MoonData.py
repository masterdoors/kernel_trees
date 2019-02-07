# coding: utf-8

'''
Created on 21 мая 2016 г.

@author: keen
'''

from sklearn import datasets
import CO2_tree as co2t
import CO2_forest as co2f
import pickle

from scipy.sparse import csr_matrix
from sklearn import preprocessing
from numpy import ndarray
from numpy import asarray
from numpy.random import randint as rint
from numpy import random
import datetime
from random import randint

from sklearn.ensemble import RandomForestClassifier

def calcAcc(Y1,Y2):
    sum_ = 0
    for i in range(len(Y1)):
        if Y1[i] == Y2[i]:
            sum_ += 1
    
    return float(sum_) / len(Y1) 

tree_deth = [3,4,5,6,7]
sratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
C = [100,500,1000,3000]
kgs = [500,1000,3000,10000]

#tree_deth = [3,4,5,6,7,8,9,10]
#sratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
#gammas = [1]

tries = 5

x,Y = datasets.make_moons(n_samples=1000, noise=0.3, random_state=0)

for i in xrange(len(Y)):
    Y[i] = Y[i] + 1

x = csr_matrix(preprocessing.normalize(x, copy=False, axis = 0))

best_v_acc = 0.

for c in C:

    for d in tree_deth:
    
        for r in sratio:
            
            for kg in kgs: 
                sum_t_acc = 0.
                sum_v_acc = 0.
                for _ in xrange(tries):
    
                    ns = random.permutation(x.shape[0])
                    x = x[ns]
                    Y = Y[ns]
    
                    x_train = x[:600,:]
                    x_validate = x[600:,:]
    
                    Y_train = Y[:600]
                    Y_validate = Y[600:]
    
                    x_sp_t = x_train
                    x_sp_v = x_validate
    
                
                    print "Test carbon forest with tree deth= ", d+1, " C= ", c, " s ratio ", r," gamma=",kg
                
                    #print  datetime.datetime.now()
    
                    #trc = co2t.CO2Tree(C=c, tol = 0.00001,max_iter=300000,kernel='polynomial',seed = randint(1, 100), max_deth=d,gamma=kg) 
                    trc = co2f.CO2_forest(C=c, tol = 0.00001,max_iter=-1,kernel='polynomial',max_deth=d,n_jobs=6,sample_ratio=r, n_estimators=10,gamma=kg)
                    #trc = RandomForestClassifier(n_estimators=100, max_depth=d, max_features=r, n_jobs=6)
    
                    trc.fit(x_sp_t, Y_train)
    
                    Y_t = trc.predict(x_sp_t)
    
                    Y_v = trc.predict(x_sp_v)
                
                    sum_t_acc += calcAcc(Y_train,Y_t)
                    sum_v_acc += calcAcc(Y_validate,Y_v)
                    #print datetime.datetime.now()
                                    
                print "Train accuracy: ", sum_t_acc / tries 
                print "Validate accuracy:", sum_v_acc / tries                 
                if sum_v_acc / tries > best_v_acc:
                    best_v_acc = sum_v_acc / tries

print best_v_acc


