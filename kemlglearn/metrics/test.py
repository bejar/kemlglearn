"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 22/12/2014 14:34 

"""

__author__ = 'bejar'

from sklearn.datasets import load_iris, make_blobs
from quantization_error import quantization_error
from Xu import Xu

#X, _ = make_blobs(n_samples=500, n_features=10, centers=5)

X= load_iris()
qe = Xu(maxc=10)

qe.fit(X['data'])



