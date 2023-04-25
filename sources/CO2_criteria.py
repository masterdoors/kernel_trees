'''
Created on Apr 25, 2023

@author: keen
'''
import numpy

def criteriaGini(pj):
    return pj*(1 - pj)

def criteriaIG(pj):
    eps = 0.0000001
    if pj == 0:
        pj += eps
    return - pj*numpy.log(pj)       

def criteriaGinirow(pj):
    return (pj*(1 - pj)).sum()

def criteriaIGrow(pj):
    eps = 0.0000001
    pj[pj == 0] = eps
    return (- pj*numpy.log(pj)).sum()  

def criteriaMSE(pj,x):
    return ((pj - x)*(pj - x)).sum()
