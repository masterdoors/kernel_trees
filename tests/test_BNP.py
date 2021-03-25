# coding: utf-8

'''
Created on 14 февр. 2018 г.

@author: keen
'''

import pandas
import numpy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import math
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
import CO2_tree as co2t
import CO2_forest as co2f
from numpy import random
from random import randint
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)


#from memory_profiler import profile

def my_func(x_mtx):
    arrs_to_conc = []       
    for i in xrange(x_mtx.shape[1]):
        arr = numpy.unique(x_mtx[:,i])

        if len(arr) < 40000:
            digitized_arr = LabelEncoder().fit_transform(x_mtx[:,i])
            if isinstance(arr[0],float) and math.isnan(arr[0]):
                nan_idx = digitized_arr == 0
                digitized_arr[nan_idx] = len(arr) * 2            
            coded_arr =  sparse.lil_matrix(OneHotEncoder(sparse=True,handle_unknown='ignore').fit_transform(digitized_arr.reshape(-1, 1)))

            arrs_to_conc.append(sparse.csr_matrix(coded_arr,dtype=float))
            #print i,coded_arr.shape

        else:
            arrs_to_conc.append(sparse.csr_matrix(x_mtx[:,i].reshape(-1, 1),dtype=float))

    return sparse.hstack(arrs_to_conc)

#@profile
def test():

    dataX = numpy.load('BNP_X.npy')
    indX = numpy.load("BNP_IndX.npy")
    ptrX = numpy.load("BNP_PtrX.npy")
    x = sparse.csr_matrix((dataX,indX,ptrX), copy=False).transpose()
    y = numpy.load("BNP_DataY.npy",mmap_mode='r')    

    print (x.shape) 
    tree_deth = [3]
    sratios = [0.0]
    fratios = [0.2]    
    C = [500]
   
    #tree_deth = [5]
    #sratios = [0.5]
    #C = [0.001]
    
    best_v_acc = 10.
    
    for d in tree_deth:
        for sratio in sratios:
            for fratio in fratios:
                for c in C:
                    t_sc = []
                    v_sc = []
                    for train, test in kf.split(x):
                        
                        print ("Test carbon forest with tree deth= ", d+1, " C= ", c, " s ratio ", sratio," f ratio ",fratio," gamma=",1)            
                        #trc = co2t.CO2Tree(C=c, tol = 0.0001,max_iter=5000000,kernel='linear',seed = randint(1, 100), feature_ratio = fratio, max_deth=d,gamma=1,dual=False)
                        trc = co2f.CO2_forest(C=c, dual=False,tol = 0.0000001,max_iter=10000,kernel='linear',max_deth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = fratio,n_estimators=10,gamma=1,dropout=sratio,noise=0.0)        
                        
                        trc.fit(sparse.csr_matrix(x[train]), y[train])

                        #weights = numpy.asarray(trc.stat()).flatten()

                        #print ("W:", numpy.histogram(weights,10,density=True))          
        
                        Y_t = trc.predict_proba(sparse.csr_matrix(x[train]),macro=True)
        
                        Y_v = trc.predict_proba(sparse.csr_matrix(x[test]),macro=True)

                        try: 
                            t_sc.append(log_loss(y[train],Y_t[:,1:]))
                            v_sc.append(log_loss(y[test],Y_v[:,1:]))                    
                            print (t_sc,v_sc)
                        except:
                            pass
                       
    
                    t_sc = numpy.asarray(t_sc)
                    v_sc = numpy.asarray(v_sc)
                    print ("Train log loss: ", t_sc.mean())
                    print ("Test log loss:", v_sc.mean())   

                     
                                  
                    if v_sc.mean() <  best_v_acc:
                        best_v_acc = v_sc.mean()                
    
    print (best_v_acc)
 
#tbl=pandas.read_csv("BNP/train.csv",sep=',')
#mtx = tbl.as_matrix()

#x_mtx = mtx[:,2:]
#y_mtx = mtx[:,1]

#y = numpy.asarray(y_mtx,dtype=int)

#for i in xrange(y.shape[0]):
#    y[i] = y[i] + 1

#res_arr = my_func(x_mtx)  
#res_arr = Imputer(strategy='median',copy=False,axis=0).fit_transform(res_arr)
#x = normalize(res_arr,axis=0) 

#numpy.save("BNP_X",x.data)
#numpy.save("BNP_IndX",x.indices)
#numpy.save("BNP_PtrX",x.indptr)
#numpy.save("BNP_DataY",y) 
    
test()    
