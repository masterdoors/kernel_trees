# coding: utf-8

'''
Created on 26 марта 2016 г.

@author: keen
'''

from numpy import random
from numpy import zeros
from numpy.random import randint

from numpy import sign
from numpy import transpose
from numpy import linalg
from scipy.sparse import csr_matrix
from numpy import int8
from numpy import float64
from numpy import int64
from numpy import exp
from numpy import multiply
from numpy import asarray

from numpy import abs

from numpy import where
from numpy import argsort

from numpy import count_nonzero
from numpy import log

from numpy import bincount
from numpy import nonzero
from scipy.sparse import coo_matrix
from numpy import intersect1d
from copy import deepcopy
from numpy import save
from numpy import float64
from numpy import unique
from numpy import absolute
from numpy import apply_along_axis



#from SVM import SVM, polynomial_kernel, gaussian_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier
#from memory_profiler import profile

#import linearSVM

from numpy import isnan
from scipy.sparse.csc import csc_matrix
  
class DecisionStamp:
    
    def estimateTetas(self,x,Y):
        counts = self.n_classes

        self.Teta0 = zeros((counts))
        self.Teta1 = zeros((counts))

        signs = self.stamp_sign(x,sample = False)

        if isinstance(signs, csr_matrix) or isinstance(signs, coo_matrix):
            signs = signs.todense()

        cl = asarray(multiply(signs,Y))

        cl = cl[nonzero(cl)]

        pos_cl = abs(cl[cl >= 0.0]).astype(int64)
        neg_cl = abs(cl[cl < 0.0]).astype(int64)

        lcount = bincount(neg_cl)
        rcount = bincount(pos_cl)

        for i in range(len(lcount)):
                self.Teta0[self.class_map[i]] += float(lcount[i])


        for i in range(len(rcount)):
                self.Teta1[self.class_map[i]] += float(rcount[i])

        
    def delta(self,H,Y):
        res = 0
        for s in (-1,+1):
            Hl = float(H[H==s].size) / H.size
            index = asarray(range(H.shape[1]))
            Hs_index = index[H[0,index] == s]
            for y in self.class_map_inv:
                y_index = index[Y[index] == y]
                common_ids = intersect1d(y_index, Hs_index) 
                if Hs_index.shape[0] != 0:
                    pj = float(common_ids.shape[0]) / Hs_index.shape[0]
                    res += Hl * pj * (1 - pj) 
 
        return res       
    
    def getDeltaParams(self,H,Y):
        res = 0
        IH = {}
        IH[-1] = float(H[H==-1].size)
        IH[1] = float(H[H==+1].size)
        Hsize = H.size
        IY = {}
        IY[-1] = {}
        IY[1] = {}
        
        for s in (-1,+1):
            Hl = float(H[H==s].size) / H.size
                
            index = asarray(range(H.shape[1]))
            Hs_index = index[H[0,index] == s]
            
            for y in self.class_map_inv:
                y_index = index[Y[index] == y]
                common_ids = intersect1d(y_index, Hs_index) 
                IY[s][y] = common_ids.shape[0]

                if Hs_index.shape[0] != 0:
                    pj = float(common_ids.shape[0]) / Hs_index.shape[0]
                    res += Hl * pj * (1 - pj) 
 
        return Hsize, IH,IY, res           
    
    def delta_wise(self, Hsize, IH,IY,yi,hi):
        res = 0
        for s in [-1,1]:
            Hl = float(IH[s] + hi*s) / Hsize
            if IH[s] + hi*s > 0:
                for y in self.class_map_inv:
                    pj = 0
                    if y == yi:
                        pj = (IY[s][y] + hi*s) / (IH[s] + hi*s)
                    else:    
                        pj = (IY[s][y]) / (IH[s] + hi*s)
                    
                    res += Hl * pj * (1 - pj)
        return res              

    def calcGiniInc(self,w,x,Y):
        signs = transpose(sign(x - w))
       
        cl = asarray(multiply(signs,Y))
        
        cl = cl[nonzero(cl)]
        
        pos_cl = abs(cl[cl >= 0.0]).astype(int64)
        neg_cl = abs(cl[cl < 0.0]).astype(int64)
        cl = abs(cl).astype(int64)
        
        #print cl
            
        gl = 0
        gr = 0
        ga = 0
        
        lcount = bincount(neg_cl)
        rcount = bincount(pos_cl)
        acount = bincount(cl)    

        for l in lcount:
            gl += (float(l)/len(neg_cl)) * (1 - float(l)/len(neg_cl))

        for r in rcount:
            gr += (float(r)/len(pos_cl)) * (1 - float(r)/len(pos_cl))
         
        for a in acount:
            ga += (float(a)/len(cl)) * (1 - float(a)/len(cl))
            
        if len(cl) > 0:     
            return  ga - (float(len(neg_cl)) / len(cl))*gl -(float(len(pos_cl)) / len(cl))*gr
        else:
            return 0.        
            
    def calcGini(self,x,Y, report = False):
        signs = self.stamp_sign(x)
        
        if isinstance(signs, csr_matrix) or isinstance(signs, coo_matrix): 
            signs = signs.todense()        
       
        cl = asarray(multiply(signs,Y))
        
        if isnan(cl).any():
            return 0.0

        cl = cl[nonzero(cl)]
        
        pos_cl = abs(cl[cl >= 0.0]).astype(int64)
        neg_cl = abs(cl[cl < 0.0]).astype(int64)
        cl = abs(cl).astype(int64)
            
        gl = 0
        gr = 0
        ga = 0
        
        lcount = bincount(neg_cl)
        rcount = bincount(pos_cl)
        acount = bincount(cl)
        
        if report:
            #print "Classes:",  acount
            #print "Classes: ", lcount
            #print "Classes: ", rcount
            #print " (",lcount.sum(),",",rcount.sum(),")"
            pass

        for l in lcount:
            gl += (float(l)/len(neg_cl)) * (1 - float(l)/len(neg_cl))

        for r in rcount:
            gr += (float(r)/len(pos_cl)) * (1 - float(r)/len(pos_cl))
         
        for a in acount:
            ga += (float(a)/len(cl)) * (1 - float(a)/len(cl))
            
        #print "Impurities: ", ga, gl, gr 
        if len(cl) > 0:     
            return  ga - (float(len(neg_cl)) / len(cl))*gl -(float(len(pos_cl)) / len(cl))*gr
        else:
            return 0.
        
    def bijection(self,w,x,i,Y,minv,maxv,eps):
        if maxv - minv > eps:
            midllev = (maxv + minv) / 2
            w_ = midllev - (midllev - minv) /2  
            l = self.calcGiniInc(w_,x,Y)              
            w_ = midllev + (maxv - midllev) /2    
            r = self.calcGiniInc(w_,x,Y)               
            if l > r:
                return self.bijection(w,x,i,Y,minv,midllev,eps) 
            else:
                return self.bijection(w,x,i,Y,midllev,maxv,eps)                     
        else:
            w.fill(0.0)
            w[0,i] = 1.0
            w[0,w.shape[1] - 1] = minv  
            return self.calcGiniInc(minv,x,Y)         
    
    
    def searchDivider(self,w,x,Y,i):
        eps = 0.0000001
        w.fill(0.0)
        x_ = asarray(x.getcol(i).todense())
        maxv = x_.max(axis = 0)[0]
        minv = x_.min(axis = 0)[0]
        
        if maxv - minv > eps:        
            return  self.bijection(w,x_,i,Y,minv,maxv,eps) 
        else:
            return 0.0 
    
    def genBestSplit(self,x,Y,q):
        
        fw = self.features_weight.todense()
        min_impurity = 0
        min_impurity_w = zeros((1, x.shape[1]))
        
        indexies = asarray(argsort(fw))[0,:]
        
        counter = self.features_weight.shape[1] - self.features_weight.nnz
        
        qm = counter + len(bincount(Y))
        
        while (counter < qm or min_impurity == 0) and counter < x.shape[1] - 1:

            i = randint(counter,x.shape[1] - 1)
            w_ = zeros((1, x.shape[1]))
            imp = self.searchDivider(w_,x,Y, indexies[i])
               
            if imp > min_impurity:
                min_impurity = imp
                min_impurity_w = w_
            
            if imp == 0:
                fw[0, indexies[i]] = 0.0    
                
            t = indexies[counter]
            indexies[counter] = indexies[i]
            indexies[i] = t                 
                
            counter += 1    
            
        self.features_weight = csr_matrix(fw)    
  
        return min_impurity_w

    def genBestSplitSerial(self,x,Y,q):
        
        fw = self.features_weight.todense()
        min_impurity = 0
        
        indexies = asarray(argsort(fw))[0,:]
        
        counter = self.features_weight.shape[1] - self.features_weight.nnz
        
        qm = counter + q
        
        imp_list = []
        
        while (counter < qm or min_impurity == 0) and counter < x.shape[1] - 1:

            i = randint(counter,x.shape[1] - 1)
            w_ = zeros((1, x.shape[1]))
            imp = self.searchDivider(w_,x,Y, indexies[i])
               
            if imp > min_impurity:
                min_impurity = imp
                imp_list.append(w_)
            
            if imp == 0:
                fw[0, indexies[i]] = 0.0    
                
            t = indexies[counter]
            indexies[counter] = indexies[i]
            indexies[i] = t                 
                
            counter += 1    
            
        self.features_weight = csr_matrix(fw)    
  
        return imp_list, min_impurity
    
    def swap_rows(self, mat, a, b):
        a_idx = where(mat.indices == a)
        b_idx = where(mat.indices == b)
        mat.indices[a_idx] = b
        mat.indices[b_idx] = a
        return mat.asformat(mat.format)                                      

    def swap_rows_batch(self, mat, a, b) :
        buf = mat[a, :]
        mat[a, :] = mat[b, :]
        mat[b, :] = buf
        return mat  
    

    #@profile
    def convexConcaveOptimization(self,x,Y,sample_weight):

        random.seed()

        if x.shape[0] > 0:
            
            sample_idx = sample_weight > 0
            
            Y_tmp = Y[sample_idx.reshape(-1)]
            x_tmp = csr_matrix(x[sample_idx.reshape(-1)])

            #sample X and Y
            if self.sample_ratio*x.shape[0] > 10:
                #idxs =  random.permutation(x_tmp.shape[0])[:int(x_tmp.shape[0]*self.sample_ratio)]            
                idxs = randint(0, x_tmp.shape[0], int(x_tmp.shape[0]*self.sample_ratio)) #bootstrap
                x_ = csr_matrix(x_tmp[idxs])
                Y_ = Y_tmp[idxs]
                    
                diff_y = unique(Y_)
                if diff_y.shape[0] > 1:
                    x_tmp = x_
                    Y_tmp = Y_
            
            def nu(arr):
                return asarray([1 + unique(arr[:,i].data,return_counts=True)[1].shape[0] for i in range(arr.shape[1])])
            
            #nonzero_idxs = unique(x_tmp.nonzero()[1]) 
            counts = nu(csc_matrix(x_tmp))
            pos_idx = where(counts > 1)[0]

            fw_size = int(x_tmp.shape[1] * self.feature_ratio)
            if fw_size > pos_idx.shape[0]:
                fw_size = pos_idx.shape[0]
            #fw_size = int(pos_idx.shape[0] * self.feature_ratio)

            self.features_weight = random.permutation(pos_idx)[:fw_size]

            x_tmp = csr_matrix(x_tmp[:,self.features_weight])

            H = zeros(shape = (1,Y_tmp.shape[0]))        
            
            gini_res = 0    
    
            class_counts = unique(Y_tmp, return_counts=True)
            class_counts = zip(class_counts[0],class_counts[1])

            class2side = {}
            class2count = {}
            side2count = {}

            for class_id, count_ in class_counts:
                left_side_prob = (count_ + side2count.get(-1,0)) / (count_ + side2count.get(-1,0) + side2count.get(1,0))
                right_side_prob =   (count_ + side2count.get(1,0)) / (count_ + side2count.get(-1,0) + side2count.get(1,0))

                left_count = side2count.get(-1,0) + count_ 
                right_count = side2count.get(1,0) + count_ 

                gini_l = 0
                gini_r = 0

                for class_ in class2side:
                    if class2side[class_] > 0:
                        gini_l += (class2count[class_] / left_count) * (1 - class2count[class_] / left_count)
                    else:
                        gini_r += (class2count[class_] / right_count) * (1 - class2count[class_] / right_count)

                gini_l += (count_   / left_count) * (1 - count_ / left_count)  
                gini_r += (count_   / right_count) * (1 - count_ / right_count)  
                class2count[class_id] = count_

                if left_side_prob * gini_l > right_side_prob * gini_r:
                    class2side[class_id] = 1
                    side2count[1] = right_count
                    H[0,Y_tmp == class_id] = 1   
                else:
                    class2side[class_id] =  -1
                    side2count[-1] = left_count      
                    H[0,Y_tmp == class_id] = -1   

            Hsize, IH,IY, gini_old_wise = self.getDeltaParams(H,Y_tmp)
                
            deltas = zeros(shape=(H.shape[1]))
            for i in range(H.shape[1]):
                gini_i = self.delta_wise(Hsize, IH,IY,Y_tmp[i],-H[0,i])
                deltas[i] = (gini_i - gini_old_wise)*H.shape[1] + 1e-12
 
            if self.kernel == 'linear':
                self.model = LinearSVC(penalty='l2',dual=self.dual,tol=self.tol,C = self.C,class_weight='balanced',\
                                       max_iter=self.max_iter)
                
                
                self.model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                
            else:
                if self.kernel == 'polynomial':
                    self.model = SVC(kernel='poly',tol=self.tol,C = self.C,class_weight='balanced',\
                       max_iter=self.max_iter,degree=3,gamma=self.gamma)
                    self.model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                else:
                    if self.kernel == 'gaussian':
                        self.model = SVC(kernel='rbf',tol=self.tol,C = self.C,class_weight='balanced',\
                           max_iter=self.max_iter,degree=3,gamma=self.gamma)
                        self.model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)

            gini_res = self.calcGini(x,Y)
            
            #print (gini_res)

            self.estimateTetas(x_tmp, Y_tmp) 

            self.p0 = zeros(shape=(self.class_max + 1))
            self.p1 = zeros(shape=(self.class_max + 1))

            sum_t0 = self.Teta0.sum()
            sum_t1 = self.Teta1.sum()

            if sum_t0 > 0: 
                p0_ = multiply(self.Teta0, 1. / sum_t0)                

                for i in range(len(p0_)):
                    self.p0[self.class_map_inv[i]] = p0_[i]

            if sum_t1 > 0:       
                p1_ = multiply(self.Teta1, 1. / sum_t1)

                for i in range(len(p1_)):
                    self.p1[self.class_map_inv[i]] = p1_[i]  
                
            return gini_res    
#public:

    def __init__(self, n_classes,class_max, features_weight, kernel='linear', \
                 sample_ratio=0.5, feature_ratio=0.5,dual=True,C=100.,tol=0.001,max_iter=1000,gamma=1000.):
        self.tol = tol
        self.n_classes = n_classes
        self.class_max = class_max
        self.features_weight = deepcopy(features_weight)
        self.C = C
        self.gamma = gamma
        self.sample_ratio = sample_ratio
        self.kernel = kernel
        self.max_iter = max_iter
        self.dual = dual
        self.feature_ratio = feature_ratio
    
    def fit(self, x,Y, sample_weight,class_map,class_map_inv):
        
        self.class_map = class_map
        self.class_map_inv = class_map_inv
        
        gres = self.convexConcaveOptimization(x,Y,sample_weight)
        x = deepcopy(x)
        
        sign_matrix_full = self.stamp_sign(x)
        sign_matrix = multiply(sample_weight.reshape(-1), sign_matrix_full)

        signs = asarray(sign_matrix)

        colsL = where(signs < 0.0)[0]
        colsR = where(signs > 0.0)[0]
        
        self.sample_weightL = zeros(shape=sample_weight.shape,dtype = int8)
        self.sample_weightL[0,colsL] = 1       
        self.sample_weightR = zeros(shape=sample_weight.shape,dtype = int8)
        self.sample_weightR[0,colsR] = 1    
        
        return gres        
    
    def stamp_sign(self,x,sample = True):
        if sample:
            x = x[:,self.features_weight]
        return sign(self.model.predict(x))
    
    def predict_proba(self,x):
        if  self.stamp_sign(x) < 0:
            return self.p0
        else:
            return self.p1
                            
