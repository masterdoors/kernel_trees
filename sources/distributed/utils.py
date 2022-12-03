import socket
import pickle
import os
import subprocess
import yaml
import numpy
import time
import signal
import uuid

import requests
import grequests
import rediswq
from tempfile import TemporaryFile

BUFFER_SIZE = 1024*128

from scipy.sparse import csr_matrix

#1 1 CMD - get task
#5 1 CMD - ping
#2 9+LEN CMD ID MASK - send task
#4 9+LEN CMD ID MASK - save result
#3 9 CMD ID - mark as in process

class BaseCmd:
    def __init__(cmd, db, res):
        self.cmd = cmd
        self.db = db
        self.res = res
    
    def __str__():
        return str(pickle.dumps(self, 0))

    def execute():
        if self.cmd == 1: 
            item = self.db.lease(lease_secs=10, block=True, timeout=2) 
            if item is not None:
                itemstr = item.decode("utf-8")
                self.db.complete(item)

class Cmd(BaseCmd):
    def __init__(cmd, mask, db, res):
        super().__init__(cmd, db, res)
        self.mask = mask

    def execute():
        if self.cmd == 2:
            try:
                self.db.push(self.mask)
            except:
                print("Data are already in the queue")
        else:
            if self.cmd == 4:
                if not self.db.empty():
                    self.db.complete(item)
                self.res.push(self.mask)

def command(cmd, id=-1, mask=None,addr=("localhost",5555)):
    
    package = [] 
    package.append(0)
    package.append(cmd)
    #print(cmd,id,addr)
    if cmd == 1 or cmd == 5:
        package[0] = int(1).to_bytes(8,byteorder='little')
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
    #if cmd == 2:
      
    #    print (cmd,len(cmd_str),9 + len(mask))
    data = []
    retry = True
    tries = 0
    while retry:
        retry = False
        tries += 1
        try:
             sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
             sock.connect(addr)
             try:
                 sock.sendall(bytes(cmd_str))
             except Exception as e:    
                 print ("CommandL error while sendall", e)
                 retry = True
                 time.sleep(1)
                 
             if cmd != 2 and cmd != 4 and cmd != 3:    
                try:  
                    msg = sock.recv(BUFFER_SIZE)
                    data = b''

                    while msg:
                        data += msg
                        msg = sock.recv(BUFFER_SIZE)
                except Exception as e:
                    print("Error while trying to read the command result. cmd: ",cmd,e)
                    retry = True
                    time.sleep(1)
             
        except Exception as e:
            print("Error while sending a command: ", cmd, e)
            retry = True
            time.sleep(1)
        finally:
            sock.close()
        if tries > 10:
            break
    if tries > 10:
        raise Exception("cmd: too many tries")
    return data  

def loadClusterCfg(cfg_path):
    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg    

def readResultFile(res_name):
    bufs = []
    offs = 8
    old_offs = 0
    if os.path.exists(res_name):
        with open(res_name,'rb') as f:
            tree_arr = f.read()
            while offs < len(tree_arr):  
                size_ = int.from_bytes(bytes(tree_arr[old_offs:offs]),byteorder='little')
                arr = bytes(tree_arr[offs :size_ + offs]) 
                id_ = int.from_bytes(bytes(arr[:8]),byteorder='little',signed=True)   
                parent_id = int.from_bytes(bytes(arr[8:16]),byteorder='little',signed=True)    
                buf = arr[16:]
                buf = pickle.loads(buf)
                bufs.append((buf,id_,parent_id))
                old_offs = offs + size_
                offs = old_offs + 8     
                
    return bufs      

def expandMatrix(x):
        x = csr_matrix((x.data, x.indices, x.indptr),shape=(x.shape[0], x.shape[1] + 1), dtype = numpy.float32, copy=False)
        tdat = [-1] * x.shape[0]
        tcols = [x.shape[1] - 1] *  x.shape[0]
        trows = range(x.shape[0])
        
        x_tmp = csr_matrix((tdat, (trows, tcols)),shape=x.shape,dtype = numpy.float32)
        
        x = x + x_tmp     
        
        return x
    
def prepareProblem(func):
    
    def run_client_cmd(fname, n_threads, docker_id):
        return "docker exec -ti " + docker_id + " /bin/bash python client.py " + fname + ' --nproc ' + str(n_threads) 
    
    def data_checker(tree,x,Y,problem,clusterCfg, res_name,addr,preprocess=False,sample_weight=None):
        if isinstance(x,csr_matrix) and isinstance(Y,numpy.ndarray):
            if Y.shape[0] > 0 and x.shape[0] == Y.shape[0]:
                
                if preprocess:
                    x = expandMatrix(x)
                
                classes_ = numpy.nonzero(numpy.bincount(Y))[0]
                n_classes = len(classes_)
                tree.class_max = Y.max()
                
                tree.class_map = numpy.zeros(shape = (tree.class_max + 1), dtype = numpy.int64)
                tree.class_map_inv = numpy.zeros(shape = (n_classes), dtype = numpy.int64)
                cc = 0
                for c in classes_:
                    tree.class_map[c] = cc
                    tree.class_map_inv[cc] = c                   
                    cc += 1 
                
                if sample_weight == None:
                    sample_weight = numpy.ones(shape=(1,x.shape[0]),dtype = numpy.int8)
                    
                id_ = str(uuid.uuid4())
                
                #problem['features'] = "/data/" + id_ + "x.npy"
                #problem['labels'] = "/data/" + id_ + "Y.npy"    
                problem['class_map'] = tree.class_map.tolist()
                problem['class_map_inv'] = tree.class_map_inv.tolist()
                problem['n_classes'] = n_classes
                problem['class_max'] = int(tree.class_max)                 
                            
                xfile_name = id_ + "_x.npy"
                yfile_name = id_ + "_y.npy"
                
                numpy.save(yfile_name,Y)
                numpy.save(xfile_name,numpy.asarray(x.todense())) 
         

                
                try:     
                    server_cmd = "./queue 5555 " + res_name + " > queue.log"
                    p = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                    print ("Run queue")
                    time.sleep(5)
                    func(tree,sample_weight,addr)                    
                    
                except Exception as e:
                    print("func():",e)
                rss = []    
                try:    
                    for srv in clusterCfg['servers']:

                        url = srv['host'] + ":" + str(srv['port']) + "/run_problem"
                        print (url)
                        for _ in range(srv['max_threads']):
                            files = {'features': open(xfile_name,'rb'), 'labels': open(yfile_name,'rb')}
                            values = {'problem':yaml.dump(problem),'id':id_}                            
                            rss.append(grequests.post(url,files=files, data=values,timeout=9000,allow_redirects=False))
                    
                    grequests.map(rss)
                    p.wait()
                    print ("Queue is stopped")
                    
                except Exception as e:
                    print("Request to the traineers:", url,e)
                    
                os.remove(xfile_name)
                os.remove(yfile_name)    
            else:
                print ("Wrong training set dimensionality")  
        else:
            print ("X type must be scipy.sparse.csr_matrix and Y type must be numpy.ndarray") 
    return data_checker       

     

    
