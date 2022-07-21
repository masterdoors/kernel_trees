# coding: utf-8

'''
Created on 26 марта 2016 г.

@author: keen
'''

import numpy

class DecisionStamp:
    def __init__(self,model,features_weight,class_max,p0,p1):
        self.model = model
        self.features_weight = features_weight
        self.class_max = class_max
        self.p0 = p0
        self.p1 = p1
    
    def stamp_sign(self,x,sample = True):
        if sample:
            x = x[:,self.features_weight]
        return numpy.sign(self.model.predict(x))

    def predict_proba(self,x,sample = True):
        res = numpy.zeros((x.shape[0],self.class_max + 1))
        sgns = self.stamp_sign(x,sample)
        res[sgns < 0] = self.p0
        res[sgns >=0] = self.p1

        return res
                            
