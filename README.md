# kernel_trees
Bagging on kernel trees

Now it uses LinearSGD solver to train oblique splits. Please build and install the patched Scikit-learn first (from the thirdparty directory) if you want to use LinearSVC instead.

sources - source files: 

--sources/cascade - adopted Deep Forest

--sources/distributed - experimental distributed implementation. 

--sources/regularization - experiments with ensemble pruning.

--sources/CO2_**.py - Kernel Forest.

tests - files and notebooks to run various experiments.

thirdparty - files to patch Scikit-learn.

# Dependences:
1. numpy/scipy 1.18.0/1.5.0 or higher

2. sympy 1.7 or higher

3. joblib 1.0.0 or higher

4. scikit-learn 0.22 or higher (use files from thirdparty if you want to utilize LinearSVC as the solver). 

# Installation:
## Kernel Forests
1. Install dependences
2. Copy sources/CO2_**.py into the working dir of your program.
## Kernel Forests with GPU acceleration
## Cascade Forests
1. Install dependences
2. Go to sources/cascade. Run setup.py install
3. Copy sources/CO2_**.py into the working dir of your program.
