import numpy
import CO2_tree as co2
from CO2_forest import * 

class GPUForest:
    def sequential_fit(self,x,Y):
        self.trees = []
        forest = self
        for _ in range(self.n_estimators):
            tree = co2.BaseCO2Tree(C=forest.C , kernel=forest.kernel,\
            tol=forest.tol, max_iter=forest.max_iter,max_deth = forest.max_depth,\
            min_samples_split = forest.min_samples_split,dual=forest.dual,\
            min_samples_leaf = forest.min_samples_leaf, seed = None,\
            sample_ratio = forest.sample_ratio, feature_ratio = forest.feature_ratio, \
            gamma=forest.gamma,criteria = forest.criteria)

            tree.fit(x,Y, preprocess = False)
            self.trees.append(tree)    
    
    def sequential_predict(self,x,use_weight=True):
        probas = []
        for tree in self.trees:
            probas.append(tree.predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  (numpy.asarray(probas).sum(axis=0), 1. / self.n_estimators)
        res =  numpy.argmax(proba, axis = 1)
        return self.le.inverse_transform(res)   
    
    def sequential_predict_proba(self,x,use_weight=True):
        probas = []
        for i in range(len(self.trees)):
            probas.append(self.trees[i].predict_proba(x,None,preprocess = False,stat_only=False,use_weight=use_weight)) 

        proba =  numpy.asarray(probas)#[:,1:]
        return proba   
    
class GPUForestClassifier(GPUForest, CO2ForestClassifier):
    pass

class GPUForestRegressor(GPUForest, CO2ForestRegressor):
    pass
    