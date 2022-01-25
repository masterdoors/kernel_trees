from thundersvm import *
from scipy.sparse.csr import csr_matrix
import numpy
import pickle
import sys

with open('shape.pickle', 'rb') as f:
    shapex = pickle.load(f)

cl_id = sys.argv[1]
kernel = sys.argv[2]
dataX = numpy.load('vertexDataX.npy')
indX = numpy.load("vertexIndX.npy")
ptrX = numpy.load("vertexPtrX.npy")
x_tmp = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32,copy=False)
H = numpy.load("DataH.npy")
deltas = numpy.load("deltas.npy")

model = SVC(kernel=kernel,tol=0.0001,C = 100,max_iter=1000,gamma=100,gpu_id=1)
model.fit(x_tmp,H.reshape(-1),sample_weights=deltas)

model.save_to_file(cl_id + ".model")
