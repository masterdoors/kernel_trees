import CO2_tree as co2t
import CO2Forest as co2f
import numpy
import pickle
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

with open("titanicX.pkl","rb") as f:
    x = pickle.load(f)

with open("titanicY.pkl","rb") as f:
    y = pickle.load(f)


y = y + 1
tree_deth = [9]
sratios = [0.1]
sratios2 = [1.0]
fratios = [0.8]

C = [5000]
#C=[5000,10000]

best_v_acc = 0.

cdp = []

for d in tree_deth:
    for sratio in sratios:
        for sratio2 in sratios2:
            for fratio in fratios:
                for c in C:
                    for ns in [0.]:
                        for cov in [1 for _ in range(10000)]:
                            v_sc = []
                            v2_sc = []
                            t_scw = []
                            t_scf = []
                            wd = []
                            
                            covs = []
                            print ("Test carbon forest with tree deth= ", d+1, " C= ", c, " s ratio ", sratio," s ration 2" ,sratio2," f ratio ",fratio," noise=",ns, "cov=",cov)
                            for train, test in kf.split(x):
        
                                trc = co2f.CO2Forest(C=c, dual=False,tol = 0.0000001,max_iter=1000000,kernel='linear',max_deth=d,n_jobs=10,sample_ratio=1.0, feature_ratio = fratio,n_estimators=10,gamma=1,dropout_low=sratio,dropout_high=sratio2,noise=ns,cov_dr=cov)
        
                                trc.fit(sparse.csr_matrix(x[train]), y[train])
        
                                #Y_v = trc.predict(sparse.csr_matrix(x[test]),True)
                                Y_v_f = trc.predict(sparse.csr_matrix(x[test]),False)
                                
                                #proba_v = trc.predict_proba(sparse.csr_matrix(x[test]),use_weight=True)
                                #proba_f = trc.predict_proba(sparse.csr_matrix(x[test]),use_weight=False)
                                
                                covs.append(numpy.abs(trc.covs).mean())
                                #for i in range(Y_v.shape[0]):
                                #    if Y_v[i] != Y_v_f[i]:
                                #        print ("diff: ")
                                #        print(proba_v[i])
                                #        print(proba_f[i])
                                
                                #wd.append(accuracy_score(Y_v,Y_v_f))
                                
                                #Y_t_w = trc.predict(sparse.csr_matrix(x[train]),True)
                                Y_t_f =  trc.predict(sparse.csr_matrix(x[train]),False)
                                #print(Y_v)
        
                                try:
                                    #v_sc.append(f1_score(y[test],Y_v))
                                    #t_scw.append(f1_score(y[train],Y_t_w))
                                    t_scf.append(f1_score(y[train],Y_t_f))
                                    v2_sc.append(f1_score(y[test],Y_v_f))
                                    
                                except Exception as e:
                                    print("Error:", e)
                            
                            print (v_sc)
                            #v_sc = numpy.asarray(v_sc)
                            v2_sc = numpy.asarray(v2_sc)
                            #t_scw = numpy.asarray(t_scw)
                            t_scf = numpy.asarray(t_scf)
                            
                            #if v_sc.mean() > best_v_acc:
                            #    best_v_acc = v_sc.mean() 
                                
                            #print ("Test f1:", v_sc.mean())
                            #print ("Delta_w: ",t_scw.mean() - v_sc.mean())
                            print ("Delta_f: ",t_scf.mean() - v2_sc.mean()) 
                            #print ("Sim: ", numpy.asarray(wd).mean())
                            print("COVS: ",numpy.asarray(covs).mean())
                            
                            for ij in range(len(t_scf)):
                                if covs[ij] > 0.38:
                                    cdp.append((t_scf[ij] - v2_sc[ij],covs[ij]))
                    
#print ("Best result is:", best_v_acc)    
print ("cov vs delta:",cdp)                

