"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 10/02/2015 9:50 

"""

__author__ = 'bejar'

from MeanPartition import MeanPartitionClustering
from kemlglearn.datasets import cluster_generator
import matplotlib.pyplot as plt


nc = 3
_, X = cluster_generator(n_clusters=nc, sepval=0.9, numNonNoisy=15, numNoisy=0, rangeN=[50, 100])


gkm = MeanPartitionClustering(n_clusters=nc, n_components=40)
res = gkm.fit(X)


fig = plt.figure()

# ax = fig.gca(projection='3d')
# pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=gkm.labels_, s=25)

ax = fig.add_subplot(111)
plt.scatter(res[:, 0], res[:, 1])

plt.show()

