# kernel_trees
Bagging on kernel trees

Now it uses LinearSGD solver to train oblique splits. Please build and install the patched Scikit-learn first (from the thirdparty directory) if you want to use LinearSVC instead.

sources - source files: 

--sources/cascade - adopted Deep Forest

--sources/distributed - experimental distributed implementation. 

--sources/regularization - experiments with ensemble pruning.

--sources/CO2_**.py - Kernel Forest.

tests - files and notebooks to run various experiments.

thirdparty - files to patch Scikit-learn (deprecated).

# Dependences:
1. numpy/scipy 1.18.0/1.5.0 or higher

2. sympy 1.7 or higher

3. joblib 1.0.0 or higher

4. Fork of the scikit-learn 0.22 (use pip install git+https://github.com/masterdoors/scikit-learn to install). 

# Installation:
## Kernel Forests
pip install git+https://github.com/masterdoors/kernel_trees.git
## Kernel Forests with GPU acceleration
1. Install the fork of ThunderSVM (It supports sample_weights): https://github.com/masterdoors/thundersvm
2. Install Kernel Forests: pip install git+https://github.com/masterdoors/kernel_trees.git
## Cascade Forests
1. Install dependences
2. Go to sources/cascade. Run setup.py install
3. Install Kernel Forests: pip install git+https://github.com/masterdoors/kernel_trees.git
