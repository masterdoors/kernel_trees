from decision_stamp import *
from gpu_optimizer import *

class GPUDecisionStampClassifier(BaseDecisionStampClassifier, GPUOptimizer):
    def optimization(self,x,Y,sample_weight,samp_counts):
        super().optimization(x,Y,sample_weight,samp_counts)
    
class GPUDecisionStampRegressor(BaseDecisionStampRegressor, GPUOptimizer):
    def optimization(self,x,Y,sample_weight,samp_counts):
        pass        
    