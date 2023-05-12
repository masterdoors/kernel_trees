'''
Created on 26 апр. 2023 г.

@author: keen
'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linprog
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline, make_pipeline

import numpy
from CO2_forest import *

def weighter(tree,forest,w, offset = 0):
    #todo: offset
    t = forest.trees[tree]
    if len(w.shape) == 1:
        t.estimateChunkWeights(w[tree])
    else:
        w_ = numpy.zeros((t.leaves_number,w.shape[0]+1))
        for i in range(1,w.shape[0]+1):
            w_[:,i] = w[i - 1,offset:offset + t.leaves_number]
        t.estimateChunkWeights(w_)        
    return t


def testerL2(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data = splits[i][0]
            lr_data_test = splits[i][1]
 
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            lr_data_test = mm.transform(lr_data_test)            

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, splits_Y[i][0])

            to_remove,before,after = tr.prune(lr_data, lr.coef_, n)
            lr_data = tr.do_prune(lr_data,to_remove)

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)                    
            lr.fit(lr_data, splits_Y[i][0])

            lr_data = tr.do_prune(lr_data_test,to_remove)
            y_pred_ = lr.predict(lr_data)                         
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))
            
    return c,n,numpy.asarray(tests).mean() + numpy.asarray(tests).std()

def testerNoise(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data_test = splits[i][1]
            lr_data = tr.addNoise(splits[i][0],n) 
            
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            lr_data_test = mm.transform(lr_data_test)               

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, splits_Y[i][0])
            y_pred_ = lr.predict(lr_data_test)                 
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))

    return c,n,numpy.asarray(tests).mean() + numpy.asarray(tests).std()             

def testerNaive(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data = splits[i][0]
            lr_data_test = splits[i][1]

            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            
            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)                    


            lr.fit(lr_data, splits_Y[i][0])
            coeffs = numpy.abs(lr.coef_)

            max_coefs = coeffs.max(axis=1)
            remain_idxs = numpy.zeros((lr.coef_.shape[1],)).astype(bool)
            for j in range(max_coefs.shape[0]):
                remain_idxs = numpy.logical_or(remain_idxs,coeffs[j] >= max_coefs[j] * n) 

            lr_data_tr = lr_data[:,remain_idxs]  
            lr = LogisticRegression(C=c,
                        fit_intercept=False,
                        solver='lbfgs',
                        max_iter=1000,
                        multi_class='multinomial', n_jobs=-1)                    
            lr.fit(lr_data_tr, splits_Y[i][0])
            lr_data = lr_data_test
            lr_data = mm.transform(lr_data)  
            
            y_pred_ = lr.predict(lr_data[:,remain_idxs])                        
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))
   
    return c,n,numpy.asarray(tests).mean() + numpy.asarray(tests).std()           

def testerL2_noise(splits,splits_Y,forests,x,Y,x_test,Y_test, c,n,n2):
    tests = []
    orig_tests = []
    for _ in range(5):
        for i in range(len(splits)): 
            tr = forests[i]
            lr_data = splits[i][0]
            lr_data_test = splits[i][1]
            lr_data = tr.addNoise(lr_data,n2) 
      
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
            lr_data_test = mm.transform(lr_data_test)               

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, splits_Y[i][0])
            
            #print("fit train",c,n,n2)

            to_remove,before,after = tr.prune(lr_data, lr.coef_, n)
            lr_data = tr.do_prune(lr_data,to_remove)

            lr = LogisticRegression(C=c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial', n_jobs=-1)                    
            lr.fit(lr_data, splits_Y[i][0])
            #print("fit test",c,n,n2)
            
            lr_data = tr.do_prune(lr_data_test,to_remove)
            y_pred_ = lr.predict(lr_data)                         
            tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))
    return c,n,n2,numpy.asarray(tests).mean() + numpy.asarray(tests).std()

class CO2ForestClassifierTested(CO2ForestClassifier):
    def    do_tests(self,x,Y,x_test, Y_test):       
            Y = self.le.transform(Y)
            Y_test = self.le.transform(Y_test)
        
            best = 1.
            best_c = 0
            best_n = 0
            best_n2 = 0
            best_r3 = 0
            best_before = 0
            best_after = 0
            
            splits = self.splits
            splits_Y = self.splits_Y
            
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerL2_noise)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n,n2) for n2 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,n2,t in tests:
                if t < best:
                    print ("New best:",t,c,n,n2)                            
                    best = t  
                    best_c = c
                    best_n = n  
                    best_n2 = n2
                     
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.addNoise(self.main_inds_train,best_n2)
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)       
             
            lr.fit(lr_data, Y) 
            
            to_remove,best_before,best_after = self.prune(lr_data, lr.coef_, best_n)
            lr_data = self.do_prune(lr_data,to_remove)            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1) 
            lr.fit(lr_data, Y)  
            #best_r3 = self.get_cov(lr_data, lr.coef_) 

            lr_data = mm.transform(self.main_inds_test)              
            
            lr_data = self.do_prune(lr_data,to_remove)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None) 
            best_f1m = f1_score(y_pred_,Y_test,average='macro')            
            print ("LR l2-norm pruning + noise test result:", best, "f1:",best_f1,best_f1m,"C:", best_c, "pruning:", best_n, "noise:", best_n2,"instability before:", best_before, "instability after:", best_after)              
            
            best = 1.
            best_c = 0
            for c in [0.01,0.1,1.0,10]:
                tests = []
                orig_tests = []
                for i in range(len(splits)): 
                    tr = self.forests[i]

                    lr_data = splits[i][0]
                    mm = make_pipeline(MinMaxScaler(), Normalizer())
                    mm.fit(lr_data)

                    lr_data = mm.transform(lr_data)                       
                    
                    lr = LogisticRegression(C=c,
                                    fit_intercept=False,
                                    solver='lbfgs',
                                    max_iter=100,
                                    multi_class='multinomial', n_jobs=-1)

                    lr.fit(lr_data, splits_Y[i][0])
                    lr_data = splits[i][1]

                    lr_data = mm.transform(lr_data)                       

                    y_pred_ = lr.predict(lr_data)                 
                    tests.append(1. - accuracy_score(splits_Y[i][1],y_pred_))

                if numpy.asarray(tests).mean() + numpy.asarray(tests).std() < best:
                    print ("New best:",numpy.asarray(tests).mean(), numpy.asarray(tests).std(), c)
                    best = numpy.asarray(tests).mean() + numpy.asarray(tests).std()
                    best_c = c
                    
            lr_data = self.main_inds_train
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)

            lr.fit(lr_data, Y)
            #best_r3 = self.get_cov(lr_data, lr.coef_)            
            lr_data = self.main_inds_test

            lr_data = mm.transform(lr_data)               
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)                
            best_f1 = f1_score(y_pred_,Y_test,average=None)
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            print ("LR orig test result:", best,"f1:",best_f1,best_f1m,"C:", best_c)            
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0                
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerNoise)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n) for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,t in tests:
                if t < best:
                    print ("New best:",t,c,n)   
                    best = t  
                    best_c = c
                    best_n = n  
                
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)
            
            lr_data = self.addNoise(self.main_inds_train, best_n)     
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            
            lr.fit(lr_data, Y)
            #est_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.main_inds_test

            lr_data = mm.transform(lr_data)               
            
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None)
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            print ("LR noised test result:", best,"f1:" ,best_f1,best_f1m,"C:", best_c, "noise:", best_n)
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerNaive)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n) for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,t in tests:
                if t < best:
                    print ("New best:",t,c,n)   
                    best = t  
                    best_c = c
                    best_n = n  
                
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.main_inds_train
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            
            lr.fit(lr_data, Y)            
            coeffs = numpy.abs(lr.coef_)

            max_coefs = coeffs.max(axis=1)
            remain_idxs = numpy.zeros((lr.coef_.shape[1],)).astype(bool)
            for i in range(max_coefs.shape[0]):
                remain_idxs = numpy.logical_or(remain_idxs,coeffs[i] >= max_coefs[i] * best_n) 
                    
            lr_data = lr_data[:,remain_idxs] 
          
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)   
            lr.fit(lr_data, Y)
            #est_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.main_inds_test


            lr_data = mm.transform(lr_data)               
            y_pred_ = lr.predict(lr_data[:,remain_idxs])    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None) 
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            print ("LR naive pruning test result:", best,"f1:",best_f1,best_f1m,"C:", best_c, "pruning:", best_n)  
            
            best = 1.
            best_c = 0
            best_n = 0
            best_r3 = 0
            best_before = 0
            best_after = 0
            tests = Parallel(n_jobs=5,backend="multiprocessing")(delayed(testerL2)(self.splits,self.splits_Y,self.forests,x,Y,x_test,Y_test,c,n) for n in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] for c in [0.01,0.1,1.0,10])            
            
            for c,n,t in tests:
                if t < best:
                    print ("New best:",t,c,n)                            
                    best = t  
                    best_c = c
                    best_n = n  

            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1)            
            
            lr_data = self.main_inds_train
            mm = make_pipeline(MinMaxScaler(), Normalizer())
            mm.fit(lr_data)

            lr_data = mm.transform(lr_data)               
            lr.fit(lr_data, Y) 
            
            to_remove,best_before,best_after = self.prune(lr_data, lr.coef_, best_n)
            lr_data = self.do_prune(lr_data,to_remove)            
            lr = LogisticRegression(C=best_c,
                            fit_intercept=False,
                            solver='lbfgs',
                            max_iter=100,
                            multi_class='multinomial', n_jobs=-1) 
            lr.fit(lr_data, Y)  
            #best_r3 = self.get_cov(lr_data, lr.coef_) 
            lr_data = self.main_inds_test
            lr_data = mm.transform(lr_data)               
            lr_data = self.do_prune(lr_data,to_remove)
            y_pred_ = lr.predict(lr_data)    
            best = 1. - accuracy_score(Y_test,y_pred_)
            best_f1 = f1_score(y_pred_,Y_test,average=None) 
            best_f1m = f1_score(y_pred_,Y_test,average='macro')
            print ("LR l2-norm pruning test result:", best,"f1:",best_f1,best_f1m,"C:", best_c, "pruning:", best_n,"instability before:", best_before, "instability after:", best_after)              
            
            self.trees = Parallel(n_jobs=5,backend="multiprocessing")(delayed( weighter)(i,self,numpy.ones(self.n_estimators,)) for i in range(self.n_estimators))
            y_pred_ = self.predict(csr_matrix(x_test))
            rs_local = accuracy_score(Y_test,y_pred_)
            y_pred_ = self.predict(x)
            rs_train = accuracy_score(Y,y_pred_)
            print ("Train error orig:", 1. - rs_train)    
            print ("Test error orig:", 1. - rs_local)    
            print ("Delta orig:", rs_train - rs_local)    
