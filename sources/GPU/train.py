from thundersvm import SVC
from scipy.sparse.csr import csr_matrix
import numpy
import pickle
import sys

# +

try:
    cl_id = sys.argv[1]
    kernel = sys.argv[2]
    gpu_id =sys.argv[3]

    with open('shapet' + cl_id + '.pickle', 'rb') as f:
        shapex = pickle.load(f)

    dataX = numpy.load('vertexDataXt' + cl_id + '.npy')
    indX = numpy.load("vertexIndXt" + cl_id + ".npy")
    ptrX = numpy.load("vertexPtrXt" + cl_id + ".npy")
    x_tmp = csr_matrix((dataX,indX,ptrX), shape=shapex,dtype=numpy.float32,copy=False)
    H = numpy.load("DataH" + cl_id + ".npy")
    deltas = numpy.load("deltas" + cl_id + ".npy")
    # -

    model = SVC(kernel=kernel,tol=0.0001,C = 5500,max_iter=10000,gamma=200,gpu_id=gpu_id)#,max_mem_size=200)
    model.fit(x_tmp,H.reshape(-1),sample_weights=deltas)

    model.save_to_file(cl_id + ".model")
    #with open(cl_id + ".model", 'wb') as handle:
    #    pickle.dump(model, handle) 
                    
except Exception as e:
    pass
    #with open(cl_id + ".model",'w') as f:
    #    f.write(str(e))
