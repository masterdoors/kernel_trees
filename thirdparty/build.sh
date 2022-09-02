mkdir sklearn
git clone git clone --branch 0.22.X  https://github.com/scikit-learn/scikit-learn.git ./sklearn
yes |cp -rf  base.py ./sklearn/scikit-learn/sklearn/svm/base.py
yes |cp -rf classes.py ./sklearn/scikit-learn/sklearn/svm/classes.py
yes |cp -rf linear.cpp sklearn/scikit-learn/sklearn/svm/src/liblinear/linear.cpp 

