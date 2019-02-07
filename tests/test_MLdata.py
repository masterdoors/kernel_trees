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
import datetime
from random import randint
import numpy


def calcAcc(Y1,Y2):
    sum_ = 0
    for i in range(len(Y1)):
        if Y1[i] == Y2[i]:
            sum_ += 1
    
    return float(sum_) / len(Y1) 

#import tempfile
#test_data_home = tempfile.mkdtemp()

#datas = datasets.fetch_mldata("mnist-original", data_home=test_data_home)

#with open("MNIST.db", 'wb') as output:
#    pickle.dump(datas, output, pickle.HIGHEST_PROTOCOL)

tree_deth = [10]
sratio = [0.8]
#tree_deth = [12,14]

tries = 1

with open("MNIST.db", 'rb') as input_:
    datas = pickle.load(input_)
    
    Y =  asarray(datas["target"]).astype('int64')

    print numpy.unique(Y,return_counts=True)

    for i in xrange(len(Y)):
        Y[i] = Y[i] + 1
    x = csr_matrix(preprocessing.normalize(datas["data"], copy=False, axis = 0))
    
    ns = rint(0,x.shape[0], size=x.shape[0])
    x = x[ns]
    Y = Y[ns]
    
    x_train = x[:60000,:]
    x_validate = x[60000:,:]
    
    Y_train = Y[:60000]
    Y_validate = Y[60000:]
    
    x_sp_t = x_train
    x_sp_v = x_validate
    
    for gamma in [5500]:

        for d in tree_deth:
        
            sum_t_acc = 0.
            sum_v_acc = 0.
            
            for r in sratio:

                sum_t_acc = 0.
                sum_v_acc = 0.
                for _ in xrange(tries):
                
                    print "Test carbon forest with tree deth= ", d+1, " gamma= ", gamma, " s ratio ", r
                
                    print  datetime.datetime.now()
    
                    trc = co2t.CO2Tree(gamma=gamma , eps = 0.005,kernel='linear',seed = randint(1, 100), max_deth=d) 
                    #trc = co2f.CO2_forest(gamma=gamma , eps = 0.005,max_features=1,max_deth=d,n_jobs=4,sample_ratio=r, n_estimators=4)
    
                    trc.fit(x_sp_t, Y_train)
    
                    Y_t = trc.predict(x_sp_t)
    
                    Y_v = trc.predict(x_sp_v)
                
                    sum_t_acc += calcAcc(Y_train,Y_t)
                    sum_v_acc += calcAcc(Y_validate,Y_v)
                    print datetime.datetime.now()
                                    
                print "Train accuracy: ", sum_t_acc / tries 
                print "Validate accuracy:", sum_v_acc / tries                 
    
    
