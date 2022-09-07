import pickle
from threading import Thread
from threading import Event
from copy import deepcopy
import numpy
import os
import yaml

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import time
import math   
from scipy import sparse 
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sympy.utilities.iterables import multiset_permutations

from flask import Flask, flash, request, redirect

import socket

BUFFER_SIZE = 1024*32

def command(cmd, id=-1, mask=None,addr=("localhost",5555)):
  
    package = [] 
    package.append(0)
    package.append(cmd)
    if cmd == 1 or cmd == 5:
        package[0] = int(cmd).to_bytes(8,byteorder='little')
    package[1] = int(cmd).to_bytes(1,byteorder='little') 
    if cmd == 2 or cmd == 4:
        if mask is not None:
            package.append(int(id).to_bytes(8,byteorder='little',signed=True))
            package.append(mask)
            package[0]  = int(9 + len(mask)).to_bytes(8,byteorder='little')
    if cmd == 3:
        package.append(int(id).to_bytes(8,byteorder='little'))
        package[0]  = int(9).to_bytes(8,byteorder='little')

    cmd_str =  b''.join(package)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(addr)
    sock.sendall(bytes(cmd_str))
    data = b''  
    try:  
        msg = sock.recv(BUFFER_SIZE)



        while msg:
            data += msg
            msg = sock.recv(BUFFER_SIZE)
    except Exception as e:
        print(cmd, str(e))
        
    sock.close() 
    return data  

def get_ping(client):
    def ping(event):
        while True:
            #print("ping",os.getpid(), self.id)
            if client.id > -1:
                command(3, id=client.id,addr=client.addr)
            time.sleep(3)
            if event.is_set():
                break        
    return ping

class Client:
    
    def calcGini(self,x,Y, model, features_weight, report = False):
        signs = self.stamp_sign(x,model,features_weight)
        
        if isinstance(signs, sparse.csr_matrix) or isinstance(signs, sparse.coo_matrix): 
            signs = signs.todense()        
       
        cl = numpy.asarray(numpy.multiply(signs,Y))
        
        if numpy.isnan(cl).any():
            return 0.0

        cl = cl[numpy.nonzero(cl)]
        
        pos_cl = abs(cl[cl >= 0.0]).astype(numpy.int64)
        neg_cl = abs(cl[cl < 0.0]).astype(numpy.int64)
        cl = abs(cl).astype(numpy.int64)
            
        gl = 0
        gr = 0
        ga = 0
        
        lcount = numpy.bincount(neg_cl)
        rcount = numpy.bincount(pos_cl)
        acount = numpy.bincount(cl)
        
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
    
    def stamp_sign(self,x,model,feature_weights=None):
        if feature_weights is not None:
            x = x[:,feature_weights]
        return numpy.sign(model.predict(x))    
    
    def estimateTetas(self,x,Y,model):
        counts = self.n_classes

        self.Teta0 = numpy.zeros((counts))
        self.Teta1 = numpy.zeros((counts))

        signs = self.stamp_sign(x,model)

        if isinstance(signs, sparse.csr_matrix) or isinstance(signs, sparse.coo_matrix):
            signs = signs.todense()

        cl = numpy.asarray(numpy.multiply(signs,Y))

        cl = cl[numpy.nonzero(cl)]

        pos_cl = abs(cl[cl >= 0.0]).astype(numpy.int64)
        neg_cl = abs(cl[cl < 0.0]).astype(numpy.int64)

        lcount = numpy.bincount(neg_cl)
        rcount = numpy.bincount(pos_cl)

        for i in range(len(lcount)):
                self.Teta0[self.class_map[i]] += float(lcount[i])


        for i in range(len(rcount)):
                self.Teta1[self.class_map[i]] += float(rcount[i])    

    def clearNode(self,Y,sample_weight):
        y = numpy.asarray(numpy.multiply(sample_weight.reshape(-1),Y))
        
        y = y[numpy.nonzero(y)]
                
        differents = numpy.unique(y).shape[0]
        
        if differents <= 1:
            return True
        else:
            return False        


    def optimization(self,x,Y,sample_weight):
        model, feature_weights = self.optimize(sample_weight)
        gini_res = self.calcGini(x,Y,model,feature_weights)

        p0 = numpy.zeros(shape=(self.class_max + 1))
        p1 = numpy.zeros(shape=(self.class_max + 1))

        sum_t0 = self.Teta0.sum()
        sum_t1 = self.Teta1.sum()

        if sum_t0 > 0: 
            p0_ = numpy.multiply(self.Teta0, 1. / sum_t0)                

            for i in range(len(p0_)):
                p0[self.class_map_inv[i]] = p0_[i]

        if sum_t1 > 0:       
            p1_ = numpy.multiply(self.Teta1, 1. / sum_t1)
            for i in range(len(p1_)):
                p1[self.class_map_inv[i]] = p1_[i]  

        return gini_res,model,feature_weights,p0,p1     

    def __init__(self, x,Y,n_classes,class_max, kernel='linear', max_depth=1,\
                 sample_ratio=0.5, feature_ratio=0.5,dual=True,C=100.,tol=0.001,max_iter=1000,gamma=1000.,intercept_scaling=1.,dropout_low=0.1,\
                 dropout_high=0.9, balance=True,noise=0.,cov_dr=0., criteria="gini",class_map={},class_map_inv={}):
        if criteria == "gain":
            self.criteria = self.criteriaIG
            self.criteria_row = self.criteriaIGrow
            self.max_criteria = 1e32
        else:
            self.criteria = self.criteriaGini 
            self.criteria_row = self.criteriaGinirow
            self.max_criteria = 1.0       
        
        self.tol = tol
        self.n_classes = n_classes
        self.class_max = class_max
        #self.features_weight = deepcopy(features_weight)
        self.C = C
        self.gamma = gamma
        self.sample_ratio = sample_ratio
        self.kernel = kernel
        self.max_iter = max_iter
        self.dual = dual
        self.feature_ratio = feature_ratio
        self.intercept_scaling = intercept_scaling
        self.dropout_low = dropout_low
        self.dropout_high = dropout_high  
        self.balance = balance
        self.noise = noise
        self.cov_dr = cov_dr 
        self.chunk_weight = 1.0 
        
        self.class_map = class_map
        self.class_map_inv = class_map_inv
        self.eps = 0.0000001
        self.x = x
        self.Y = Y

        print(numpy.unique(Y))
        self.addr = ("keen.isa.ru",5555) 
        self.id = -1
        self.max_depth = max_depth
        self.stop = False


    def run_node(self):
        #thread_proc = Thread(target = self.fit, name="proc")
        thread_ping = Thread(target = ping, name="ping")
        #thread_proc.start()
        thread_ping.start()
        #thread_ping.join()

    def fit(self):
        while True:
            try:
                print ("fit", self.addr)
                data = command(5,addr=self.addr)
                arr_data = bytearray(data)
                q_idle = int.from_bytes(bytes(arr_data[:8]),byteorder='little',signed=True)
                q_run = int.from_bytes(bytes(arr_data[8:]),byteorder='little',signed=True)
                print("Check the queue status (idle/run):",arr_data,q_idle,q_run)
                
                data = command(1,addr=self.addr)
                if len(data) > 0:
                    ###get data to process
                    arr_data = bytearray(data)
                    size = int.from_bytes(bytes(arr_data[:8]),byteorder='little')
                    self.id = int.from_bytes(bytes(arr_data[8:16]),byteorder='little',signed=True)
                    command(3, id=self.id,addr=self.addr)
                    parId = int.from_bytes(bytes(arr_data[16:24]),byteorder='little',signed=True)
                    depth = int.from_bytes(bytes(arr_data[24:25]),byteorder='little')
                    side =  int.from_bytes(bytes(arr_data[25:26]),byteorder='little')
                    sample_weight = pickle.loads(bytes(arr_data[26:]))  
                    ####
                   
                    print("Run optimization",os.getpid(), self.id)
                    gini_res,model,feature_weights,p0,p1   = self.optimization(self.x,self.Y,sample_weight)
                    print("Gini res",os.getpid(), gini_res)
                    if gini_res > 0: 
                        
                        print (os.getpid(),depth,self.max_depth)
            
                        if depth + 1 < self.max_depth:        
                            sample_weightL = numpy.zeros(shape=sample_weight.shape,dtype = numpy.int8)
                            sample_weightR = numpy.zeros(shape=sample_weight.shape,dtype = numpy.int8)
            
                            sign_matrix_full = self.stamp_sign(self.x,model,feature_weights)
                            sign_matrix = numpy.multiply(sample_weight.reshape(-1), sign_matrix_full)
                            signs = numpy.asarray(sign_matrix)
                            colsL = numpy.where(signs < 0.0)[0]
                            colsR = numpy.where(signs > 0.0)[0]
                            sample_weightL[0,colsL] = 1       
                            sample_weightR[0,colsR] = 1  
                            if not self.clearNode(self.Y, sample_weightL) and numpy.count_nonzero(sample_weightL) > 1: 
                                command(2,self.id,mask= (depth + 1).to_bytes(1,byteorder='little') + int(0).to_bytes(1,byteorder='little') + pickle.dumps(sample_weightL) ,addr=self.addr)  
                                print (os.getpid(),"Load left task")
                            if not self.clearNode(self.Y, sample_weightR) and numpy.count_nonzero(sample_weightR) > 1: 
                                command(2,self.id,mask= (depth + 1).to_bytes(1,byteorder='little') + int(1).to_bytes(1,byteorder='little') + pickle.dumps(sample_weightR),addr=self.addr)
                                print (os.getpid(),"Load right task")
                                
                    print ("Mark ", os.getpid(),self.id, " as done...")
                    command(4, self.id,pickle.dumps((model,feature_weights,p0,p1,side,self.class_max)),addr=self.addr)
                else:
                    data = command(5,addr=self.addr)    
                    arr_data = bytearray(data)
                    q_idle = int.from_bytes(bytes(arr_data[:8]),byteorder='little')
                    q_run = int.from_bytes(bytes(arr_data[8:]),byteorder='little')
                    print("Check the queue status (idle/run):",q_idle,q_run)
                    if q_idle == 0 and q_run == 0:
                        self.id = -1
                        break
            except Exception as e:
                print("Exception is the main cycle: ",e)    
            finally:        
                self.id = -1 

      
    def criteriaGini(self,pj):
        return pj*(1 - pj)
     
    def criteriaIG(self,pj):
        if pj == 0:
            pj += self.eps
        return - pj*numpy.log(pj)       
     
    def criteriaGinirow(self,pj):
        return (pj*(1 - pj)).sum()
     
    def criteriaIGrow(self,pj):
        pj[pj == 0] = self.eps
        return (- pj*numpy.log(pj)).sum()  

    def getDeltaParams(self,H,Y,criteria):
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
            index = numpy.asarray(range(H.shape[1]))
            Hs_index = index[H[0,index] == s]
      
            for y in self.class_map_inv:
                y_index = index[Y[index] == y]
                common_ids = numpy.intersect1d(y_index, Hs_index) 
                IY[s][y] = common_ids.shape[0]

                if Hs_index.shape[0] != 0:
                    pj = float(common_ids.shape[0]) / Hs_index.shape[0]
                    res += Hl *  criteria(pj) 
  
        return Hsize, IH,IY, res           
     
    def delta_wise(self, Hsize, IH,IY,yi,hi, criteria):
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
           
                    res += Hl * criteria(pj)
            if math.isnan(res):
                res = 0
        return res      

    def optimize(self, sample_weight):
        print ("Start training...",os.getpid(),self.id,numpy.count_nonzero(sample_weight),numpy.count_nonzero(self.x),numpy.count_nonzero(self.Y))

        numpy.random.seed()
        if self.x.shape[0] > 0:
            sample_idx = sample_weight > 0
            sample_idx_ran = numpy.asarray(range(self.x.shape[0]))[sample_idx.reshape(-1)]
      
            Y_tmp = self.Y[sample_idx.reshape(-1)].flatten()

            x_tmp = sparse.csr_matrix(self.x[sample_idx.reshape(-1)],dtype=numpy.float32)

            #sample X and Y
            if self.sample_ratio*self.x.shape[0] > 10:
                idxs = numpy.random.randint(0, x_tmp.shape[0], int(x_tmp.shape[0]*self.sample_ratio)) #bootstrap
                x_ = sparse.csr_matrix(x_tmp[idxs],dtype=numpy.float32)
                Y_ = Y_tmp[idxs]
                diff_y = numpy.unique(Y_)
            if diff_y.shape[0] > 1:
                x_tmp = x_
                Y_tmp = Y_

        def nu(arr):
            return numpy.asarray([1 + numpy.unique(arr[:,i].data,return_counts=True)[1].shape[0] for i in range(arr.shape[1])])
      
        counts_p = nu(sparse.csc_matrix(x_tmp))
        pos_idx = numpy.where(counts_p > 1)[0]

        fw_size = int(x_tmp.shape[1] * self.feature_ratio)
        if fw_size > pos_idx.shape[0]:
            fw_size = pos_idx.shape[0]

        self.features_weight = numpy.random.permutation(pos_idx)[:fw_size]
      
        if fw_size == 0:
            return 0.

        x_tmp = sparse.csr_matrix(x_tmp[:,self.features_weight],dtype=numpy.float32)
        H = numpy.zeros(shape = (1,Y_tmp.shape[0]))        
        gini_res = 0    
        class_counts = numpy.unique(Y_tmp, return_counts=True)
        class_counts = numpy.asarray(list(zip(class_counts[0],class_counts[1])))

        class2side = {}
        class2count = {}
        side2count = {}

        min_gini = self.max_criteria
        min_p = []
        print ("Set classes to side")

        if len(class_counts) > 13:
        #Greedy
            print (class_counts)
            for _ in range(len(class_counts)*len(class_counts)*15):
                lmin_gini = self.max_criteria
                lmin_p = []
                next = True
                elements = [-1,+1]
                probabilities = [0.5, 0.5]
                p = numpy.random.choice(elements,len(class_counts) , p=probabilities)

                zc = 0 
                while next:
                    
                    next = False
                    zc += 1  
                    for i in range(p.shape[0]):
                        p[i] = - p[i]
                        left_counts = class_counts[p < 0, 1]
                        right_counts = class_counts[p > 0, 1]

                        lcs = left_counts.sum()
                        rcs = right_counts.sum()  
                        den = lcs + rcs 

                        PL = float(lcs)/ den
                        PR = float(rcs)/ den
      
                        gini_l = self.criteria_row(left_counts / lcs)
                        gini_r = self.criteria_row(right_counts / rcs)

                        gini =  PL*gini_l + PR* gini_r
                        if gini < lmin_gini:
                            lmin_p = deepcopy(p)
                            lmin_gini = gini
                            next = True
                    p = lmin_p

                if  lmin_gini < min_gini:
                    min_p = deepcopy(lmin_p)
                    min_gini = lmin_gini
  
        else:
            for zc in range(1,len(class_counts),1):
         
                a = numpy.hstack([-numpy.ones((zc,)),numpy.ones((len(class_counts) - zc,))])
                for p in multiset_permutations(a):
                    
                    p = numpy.asarray(p)
                    left_counts = class_counts[p < 0, 1]
                    right_counts = class_counts[p > 0, 1]
                    lcs = left_counts.sum()
                    rcs = right_counts.sum()  
                    den = lcs + rcs 

                    PL = float(lcs)/ den
                    PR = float(rcs)/ den
      
                    gini_l = self.criteria_row(left_counts / lcs)
                    gini_r = self.criteria_row(right_counts / rcs)

                    gini =  PL*gini_l + PR* gini_r

                    if gini < min_gini:
                        min_p = p
                        min_gini = gini

        left_counts = numpy.asarray([c[1] for c in class_counts[min_p < 0]])
        right_counts = numpy.asarray([c[1] for c in class_counts[min_p > 0]])
        side2count[-1] = left_counts.sum()
        side2count[1] = right_counts.sum()               
        for i,(cl,cnt) in enumerate(class_counts):
            class2side[cl] = min_p[i]
            H[0,Y_tmp == cl] = min_p[i]     
            class2count[cl] = cnt

        gini_best = 0
        gini_old = 0
        for class_id, count_ in class_counts:
            p = float(count_) / side2count[class2side[class_id]]
            p2 = float(count_) / (side2count[-1] + side2count[1])

            gini_old += self.criteria(p2)
            gini_best +=  (float(side2count[class2side[class_id]])/ (side2count[-1] + side2count[1]))*self.criteria(p)

        Hsize, IH,IY, gini_old_wise = self.getDeltaParams(H,Y_tmp, self.criteria)
        gini_best = gini_old - gini_best
        
        print ("Set sample weight")   
        deltas = numpy.zeros(shape=(H.shape[1]))
        for i in range(H.shape[1]):
            gini_i = self.delta_wise(Hsize, IH,IY,Y_tmp[i],-H[0,i],self.criteria)
            deltas[i] = float(gini_i - gini_old_wise)  

            if self.balance:
                deltas[i] = deltas[i] * float(H.reshape(-1).shape[0]) / (2*side2count[H[0,i]])

        ratio = 1

        dm = deltas.max()
        if deltas.max() == 0:
            deltas = numpy.ones(shape=(H.shape[1]))  
        else:
            deltas = (deltas / dm)*ratio 

        if self.noise > 0.:
            gauss_noise = numpy.random.normal(numpy.ones((x_tmp.shape[1],),dtype=float),self.noise,(1,x_tmp.shape[1]))
            x_tmp = sparse.csr_matrix(x_tmp.multiply(gauss_noise),dtype=numpy.float32)
        try:
            print ("Train a classifier",self.kernel)
            if self.kernel == 'linear':
                if not self.dual:
                    model = SGDClassifier(n_iter_no_change=5,loss='squared_hinge', alpha=1. / (100*self.C), fit_intercept=True, max_iter=self.max_iter, tol=self.tol, eta0=0.5,shuffle=True, learning_rate='adaptive')
                    #self.model = LinearSVC(penalty='l2',dual=self.dual,tol=self.tol,C = self.C,max_iter=self.max_iter)
                    model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                else:  
                    model = LinearSVC(penalty='l2',dual=self.dual,tol=self.tol,C = self.C,max_iter=self.max_iter)
                    model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
            else:
                if self.kernel == 'polynomial':
                    model = SVC(kernel='poly',tol=self.tol,C = self.C,max_iter=self.max_iter,degree=3,gamma=self.gamma)
                    model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
                else:
                    if self.kernel == 'gaussian':
                        model = SVC(kernel='rbf',tol=self.tol,C = self.C,max_iter=self.max_iter)
                        model.fit(x_tmp,H.reshape(-1),sample_weight=deltas)
        except:
            print("Error ",self.id)   
            return 0.            

        self.estimateTetas(x_tmp, Y_tmp,model)    
        print("Done",os.getpid(),self.id)             
        return model,self.features_weight

UPLOAD_FOLDER = '/data'
ALLOWED_EXTENSIONS = {'npy','yml'}

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/run_problem', methods=['GET', 'POST'])
def run_problem():
    if request.method == 'POST':
        if 'features' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if 'labels' not in request.files:
            flash('No file part')
            return redirect(request.url)   
             
        features = request.files['features']
        labels = request.files['labels']

        if features:
            x = numpy.load(features)

        if labels:
            Y = numpy.load(labels)
            
        cfg = request.values['problem']
        cfg = yaml.safe_load(cfg)
    
        kernel = cfg.get('kernel','linear') 
        dual = cfg.get('dual',False)
        C = cfg.get('C',100)
        tol = cfg.get('tol',0.0001)
        max_iter = cfg.get('max_iter',10000)
        criteria = cfg.get('criteria','gini')
        balance = cfg.get('balance',True)
        sample_ratio = cfg.get('sample_ratio',1.0)
        feature_ratio = cfg.get('feature_ratio',0.5)
        dropout_low = cfg.get('dropout_low',0.)
        dropout_high = cfg.get('dropout_high',1.0)
        class_map = numpy.asarray(cfg['class_map'])
        class_map_inv =  numpy.asarray(cfg['class_map_inv'])
        max_depth = cfg.get('max_depth',1)
        n_classes = cfg.get('n_classes',2)
        class_max = cfg.get('class_max',1)

        client = Client(x,Y,n_classes,class_max, kernel=kernel, max_depth=max_depth,\
                     sample_ratio=sample_ratio, feature_ratio=feature_ratio,dual=dual,C=C,tol=tol,max_iter=max_iter,dropout_low=dropout_low, \
                     dropout_high=dropout_high, balance=balance, criteria=criteria,class_map=class_map,class_map_inv=class_map_inv)
        try:
            event = Event()
            thread_ping = Thread(target = get_ping(client), name="ping",args=(event,))
            thread_ping.start()
                      
            client.fit()
            event.set()
            print ("Fit is finished. Wait for the ping thread...")
            thread_ping.join()
            print("The ping thread is finished. Exit run_problem.")
        except Exception as e:
            print("The utils has been finished with the message:", e)
                
            return "OK"
        return "Problem not found"            

if __name__ == '__main__':
     application.run(host='0.0.0.0',port=80)

