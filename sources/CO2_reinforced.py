from CO2_forest import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from virtualenv.create.via_global_ref.builtin import via_global_self_do

class Reinforced:
    def addNoise(self,indicators, noise= 0.):
        indicators = deepcopy(indicators)
        for i in range(indicators.shape[1]):
            #print (indicators[:,i])
            nonzero = numpy.where(indicators[:,i] > 0)[0]
            idxs = numpy.random.randint(0, nonzero.shape[0], int(nonzero.shape[0]*noise)) 
            indicators[nonzero[idxs],i] = 0. #let's make the classifiers different again
        return indicators 
        
    def prune(self, indicators, coefs, ratio):
        offset = 0
        j = 0
        norm_sums = {}
        for i in range(len(self.trees)):
            while j - offset < self.trees[i].leaves_number:
                if j - offset in self.trees[i].adj_leaves:
                    v1 = numpy.asarray([coefs[k,j] for k in range(coefs.shape[0])])
                    v2 = numpy.asarray([coefs[k,j + 1] for k in range(coefs.shape[0])])
                    norm_sums[j] = numpy.linalg.norm(v1) + numpy.linalg.norm(v2)
                j += 1
            offset += self.trees[i].leaves_number        
        to_remove = dict(sorted(norm_sums.items(), key=lambda kv: kv[1])[:int(len(norm_sums)*ratio)])
        j = 0
        offset = 0
        pr_before = 0
        pr_after = 0
        for i in range(len(self.trees)):
            while j - offset < self.trees[i].leaves_number:
                if (j - offset in self.trees[i].adj_leaves) and (j in to_remove):
                    pr_before += sum(self.trees[i].adj_leaves[j - offset])
                    pr_after += self.trees[i].adj_leaves[j - offset][0]
                j += 1
            offset += self.trees[i].leaves_number          
 
        return to_remove, pr_before / len(self.trees),pr_after / len(self.trees) 
    
    def do_prune(self, indicators, to_remove):
        for i in to_remove:
            indicators[:,i] = indicators[:,i] + indicators[:,i +1]
        indicators = numpy.delete(indicators, [idx + 1 for idx in to_remove], axis=1)
        indicators[indicators > 1.] = 1.   
        return indicators  
    
    def reinforce_prune(self,n,C,X,Y,sample_weigths = None):
        lr = LogisticRegression(C=C,
                        fit_intercept=False,
                        solver='lbfgs',
                        max_iter=100,
                        multi_class='multinomial', n_jobs=-1)            

        lr_data = self.getIndicators(X)
        mm = make_pipeline(MinMaxScaler(), Normalizer())
        mm.fit(lr_data)

        lr_data = mm.transform(lr_data)               
        lr.fit(lr_data, Y) 

        to_remove,_,_ = self.prune(lr_data, lr.coef_, n)
        lr_data = self.do_prune(lr_data,to_remove) 
        self.to_remove = to_remove
        lr = LogisticRegression(C=C,
                        fit_intercept=False,
                        solver='lbfgs',
                        max_iter=100,
                        multi_class='multinomial', n_jobs=-1) 
        lr.fit(lr_data, Y,sample_weigths) 
        self.lr = lr
        
class ReinforcedForestClassifier(CO2ForestClassifier, Reinforced):
     def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='gini',spatial_mul=1.0, id_=0,univariate_ratio=0.0,verbose=0):
         pass
     
     def fit(self):
         pass
     
     def predict(self):
         pass

class ReinforcedForestRegressor(CO2ForestRegressor, Reinforced):
     def __init__(self,C, kernel = 'linear', max_depth = None, tol = 0.001, min_samples_split = 2, \
                 dual=True,max_iter=1000000,
                 min_samples_leaf = 1, n_jobs=1, n_estimators = 10,sample_ratio = 1.0,feature_ratio=1.0,\
                 gamma=1000.,criteria='mse',spatial_mul=1.0, id_=0,univariate_ratio=0.0, verbose=0):
         pass
    
     def fit(self):
         pass
     
     def predict(self):
         pass
    
    