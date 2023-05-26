from decision_stamp import *
from gpu_optimizer import *

from scipy.sparse import csr_matrix

class GPUDecisionStampClassifier(BaseDecisionStampClassifier, GPUOptimizer):
    def stamp_sign(self,x,train_data, sample = True):
        if sample:
            res = self.model.predict(x[:,self.features_weight], csr_matrix(train_data[self.sample_weight][:,self.features_weight]))
            return np.sign(res)
        else:
            res = self.model.predict(x, train_data[self.sample_weight][:,self.features_weight])
            return np.sign(res) 

    def optimization(self,x,Y,sample_weight,samp_counts):
        super().optimization(x,Y,sample_weight,samp_counts)
    
class GPUDecisionStampRegressor(BaseDecisionStampRegressor, GPUOptimizer):
    def stamp_sign(self,x,train_data, sample = True):
        if sample:
            res = self.model.predict(x[:,self.features_weight], csr_matrix(train_data[self.sample_weight][:,self.features_weight]))
            return np.sign(res)
        else:
            res = self.model.predict(x, train_data[self.sample_weight][:,self.features_weight])
        return np.sign(res) 

    def optimization(self,x,Y,sample_weight,samp_counts):
        super().optimization(x,Y,sample_weight,samp_counts)       
    
