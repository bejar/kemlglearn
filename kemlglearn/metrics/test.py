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
from cluster import jeffrey_divergence_score
from sklearn.cluster import KMeans
from kemlglearn.datasets import cluster_generator

nc = 5
X, y = cluster_generator(n_clusters=nc, sepval=0.2, numNonNoisy=20, numNoisy=3, rangeN=[150, 200])

#X, _ = make_blobs(n_samples=500, n_features=10, centers=5)

#X= load_iris()['data']

for nclust in range(2,10):
    km = KMeans(n_clusters=nclust)
    km.fit(X)
    v = jeffrey_divergence_score(X, km.labels_)
    print(nclust, v)

#qe.fit(X['data'])



