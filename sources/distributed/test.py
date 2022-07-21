'''
Created on 20 апр. 2021 г.

@author: keen
'''

import pickle
from CO2_forest import CO2_forest
from scipy import  sparse
from sklearn.metrics import accuracy_score

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


with open("titanicX.pkl","rb") as f:
    x = pickle.load(f)

with open("titanicY.pkl","rb") as f:
    y = pickle.load(f)
    
y = y + 1    
forest = CO2_forest(C=1000, dual=False,tol = 0.0000001,max_iter=1000000,cluster_cfg='servers.yml',
                    kernel='linear',max_deth=6,n_jobs=10,sample_ratio=1.0, 
                    feature_ratio = 0.8,n_estimators=100,gamma=1,dropout_low=0,dropout_high=1,criteria='gain')

print ("Fit")
forest.fit(sparse.csr_matrix(x),y)

print("Predict accuracy:")
print(accuracy_score(y,forest.predict(sparse.csr_matrix(x))))
    
    
