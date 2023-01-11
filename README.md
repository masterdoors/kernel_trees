# kernel_trees
Bagging on kernel trees

Now it uses LinearSGD solver to train oblique splits. Please build and install the patched Scikit-learn first (from the thirdparty directory) if you want to use LinearSCV instead.

sources - source files. 

--sources/cascade - adopted Deep Forest

--sources/distributed - experimental distributed implementation. 

--sources/regularization - experiments with ensemble pruning.

--sources/CO2_**.py - Kernel Forest.

tests - files and notebooks to run various experiments.

thirdparty - files to patch Scikit-learn.
