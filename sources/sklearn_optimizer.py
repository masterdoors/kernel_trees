import numpy as np
import numpy

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

import traceback

class SKLearnOptimizer:
    def optimization(self,x,Y,sample_weight,samp_counts):
        #random.seed()
        self.counts = numpy.zeros((x.shape[0],))
        if x.shape[0] > 0:
            sample_idx = sample_weight > 0
            sample_idx_ran = np.asarray(range(x.shape[0]))[sample_idx.reshape(-1)]
            Y_tmp = Y[sample_idx.reshape(-1)]
            #x_tmp = csr_matrix(x[sample_idx.reshape(-1)],dtype=np.float32)
            rng = np.random.default_rng(self.seed)#+abs(int(x_tmp.sum())))

            #sample X and Y
            if self.sample_ratio*x.shape[0] > 10:
                #idxs =  random.permutation(x_tmp.shape[0])[:int(x_tmp.shape[0]*self.sample_ratio)]   
                idxs = rng.integers(0, sample_idx_ran.shape[0], int(sample_idx_ran.shape[0]*self.sample_ratio)) #bootstrap
                  
                to_add_cnt = numpy.unique(sample_idx_ran[idxs]) 
                #x_ = csr_matrix(x_tmp[idxs],dtype=np.float32)
                Y_ = Y_tmp[idxs]
                    
                diff_y = np.unique(Y_)
                if diff_y.shape[0] > 1:
                    #x_tmp = x_
                    sample_idx_ran = sample_idx_ran[idxs]
                    Y_tmp = Y_
                    #print ("sampling shape:",diff_y.shape[0])
            else:
                to_add_cnt = sample_idx_ran

            if not (samp_counts is None): 
                self.counts[to_add_cnt] += 1
            
            def nu(arr):
                return np.asarray([1 + np.unique(arr[:,i].data,return_counts=True)[1].shape[0] for i in range(arr.shape[1])])
            
            if isinstance(x,csr_matrix):
                counts_p = nu(csc_matrix(x[sample_idx_ran]))
            else:
                counts_p = nu(x[sample_idx_ran])
                    
            pos_idx = np.where(counts_p > 1)[0]
            
            if self.features_weight is not None:
                f_idx = np.where(self.features_weight > 0)[0]
                pos_idx =list(set(pos_idx).intersection(set(f_idx)))
                fw_size = int(self.features_weight[self.features_weight > 0].shape[0]* self.feature_ratio)
            else:
                fw_size = int(x.shape[1] * self.feature_ratio)
                if fw_size > pos_idx.shape[0]:
                    fw_size = pos_idx.shape[0]
                #fw_size = int(pos_idx.shape[0] * self.feature_ratio)
            if fw_size == 0:
                fw_size = 1
            
            self.features_weight = rng.permutation(pos_idx)[:fw_size]#.astype(int8)

            self.sample_weight = sample_idx_ran 

            H, deltas = self.setupSlackRescaling(Y_tmp)
                
            try:
                if self.kernel == 'linear':
                    if not self.dual:
                        self.model = SGDClassifier(n_iter_no_change=5,loss='squared_hinge', alpha=1. / (100*self.C), fit_intercept=True, max_iter=self.max_iter, tol=self.tol, eta0=0.5,shuffle=True, learning_rate='adaptive')
                        self.model.fit(x[sample_idx_ran][:, self.features_weight],H.reshape(-1),sample_weight=deltas)
                    else:  
                        self.model = LinearSVC(penalty='l2',dual=self.dual,tol=self.tol,C = self.C,max_iter=self.max_iter, verbose=2)
                        self.model.fit(x[sample_idx_ran][:, self.features_weight],H.reshape(-1),sample_weight=deltas)
                    
                #else:
                if self.kernel == 'polynomial':
                    self.model = SVC(kernel='poly',tol=self.tol,C = self.C,max_iter=self.max_iter,degree=4,gamma=self.gamma)
                    self.model.fit(x,H.reshape(-1),self.features_weight,sample_idx_ran,sample_weight=deltas)   
                else:
                    if self.kernel == 'gaussian':
                        self.model = SVC(kernel='rbf',tol=self.tol,C = self.C,max_iter=self.max_iter,gamma=self.gamma,cache_size=4000)
                        self.model.fit(x,H.reshape(-1),self.features_weight,sample_idx_ran,sample_weight=deltas)   
                    else:
                        if self.kernel == 'univariate':
                            self.model = DecisionTreeClassifier( criterion= self.criteria_str, max_depth=1)
                            self.model.fit(x[sample_idx_ran][:, self.features_weight],H.reshape(-1))
                        
            except Exception as exp:
                print (str(exp))
                print(traceback.format_exc())
                return 0.            

            gini_res = self.estimateOutput(x,Y_tmp)  
            return gini_res    
    
    