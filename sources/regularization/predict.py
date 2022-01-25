from thundersvm import *
from scipy.sparse.csr import csr_matrix
import numpy
import pickle
import sys

with open('shape.pickle', 'rb') as f:
    shapex = pickle.load(f)

cl_id = sys.argv[1]
dataX = numpy.load('vertexDataX.npy')
indX = numpy.load("vertexIndX.npy")
ptrX = numpy.load("vertexPtrX.npy")
x = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32,copy=False)

model = SVC(kernel='rbf',tol=0.0001,C = 2000,max_iter=1000,gamma=10,gpu_id=1)

model.load_from_file(cl_id + ".model")

res = model.predict(x)

numpy.save("sign",res)   

