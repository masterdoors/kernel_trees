#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
from os.path import dirname
from os.path import join

from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='kernel_forest',
    version='0.0.1',
    description='Oblique and kernel trees library',
    author='ISA RAS',
    author_email='',
    packages=[''],
    include_package_data=True,
    zip_safe=False,
    package_dir={'': 'sources'},
    install_requires=["Cython",'numpy==1.20.1','scipy','sympy','joblib','scikit-learn @ git+https://github.com/masterdoors/scikit-learn#egg=scikit-learn=0.24.2','optuna']
)
