'''
Created on 16 мар. 2021 г.

@author: keen
'''
# coding: utf-8

from sklearn import datasets
import pickle

from scipy.sparse import csr_matrix
from sklearn import preprocessing
from numpy import ndarray
from numpy import asarray
from numpy.random import randint as rint
import datetime
from random import randint
import numpy

import CO2_tree as co2t
import CO2Forest as co2f


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


sratio = [0.01,0.02,0.04,0.08,0.16,0.32]
fratio = [0.05, 0.08,0.1,0.2,0.3]
tree_deth = [4]

tries = 3

with open("MNIST.db", 'rb') as input_:
    datas = pickle.load(input_,encoding='latin1')

    Y =  asarray(datas["target"]).astype('int64')

    print (numpy.unique(Y,return_counts=True))

    for i in range(len(Y)):
        Y[i] = Y[i] + 1
    x = preprocessing.normalize(datas["data"], copy=False, axis = 0)

    #x = x.reshape(-1,28,28)
    
     
    ns = rint(0,x.shape[0], size=x.shape[0])
    x = x[ns]
    Y = Y[ns]

    x_train = x[:60000,:]
    x_validate = x[60000:,:]

    Y_train = Y[:60000]#[:6000]
    Y_validate = Y[60000:]#[:3000]

    x_sp_t = csr_matrix(x_train,dtype=numpy.float32)#[:6000]
    x_sp_v = csr_matrix(x_validate,dtype=numpy.float32)#[:3000]

    for C in [5500]:

        for d in tree_deth:

            sum_t_acc = 0.
            sum_v_acc = 0.

            for r in sratio:
                for f in fratio: 
                    #for l in [2]:
                    sum_t_acc = 0.
                    sum_v_acc = 0.
                    for _ in range(tries):
 
                        print ("Test carbon rbf forest with tree deth= ", d+1, " C= ", C, " s ratio ", r," f ratio ",f)#,"l:",l)

                        print  (datetime.datetime.now())
                        trc = co2f.CO2Forest(C=C, dual=False,tol = 0.0000001,max_iter=1000000,kernel='gaussian',max_deth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = f,n_estimators=10,gamma=1,dropout_low=r,dropout_high=1.,noise=0,cov_dr=0,criteria='gain')
                        
                        trc.fit(x_sp_t, Y_train)

                        Y_t = trc.predict(x_sp_t)

                        Y_v = trc.predict(x_sp_v)

                        sum_t_acc += calcAcc(Y_train,Y_t)
                        sum_v_acc += calcAcc(Y_validate,Y_v)
                        print (datetime.datetime.now())
    
                        print ("Train accuracy: ", sum_t_acc / tries)
                        print ("Validate accuracy:", sum_v_acc / tries)

