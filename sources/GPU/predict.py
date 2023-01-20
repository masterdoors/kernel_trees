from sklearn.svm import SVC
from scipy.sparse.csr import csr_matrix
import numpy
import pickle
import sys

# +
cl_id = sys.argv[1]
cur_id = sys.argv[2]
gpu_id =sys.argv[3]

with open('shape' + cur_id + '.pickle', 'rb') as f:
    shapex = pickle.load(f)

dataX = numpy.load('vertexDataX' + cur_id + '.npy')
indX = numpy.load("vertexIndX" + cur_id + ".npy")
ptrX = numpy.load("vertexPtrX" + cur_id + ".npy")
x = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32,copy=False)


model = SVC(kernel='rbf',tol=0.0001,C = 3000,max_iter=1000,gamma=100,gpu_id=gpu_id)


model.load_from_file(cl_id + ".model")


res = model.predict(x)

numpy.save("sign" + cur_id,res)   

