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

from sklearn.datasets import make_blobs, load_iris, make_circles

nc = 15
X, y = cluster_generator(n_clusters=nc, sepval=0.2, numNonNoisy=100, numNoisy=10, rangeN=[150, 200])

# print X[:,:-2]
#
#print X,y


# X, y = make_blobs(n_samples=1000, n_features=20, centers=nc, random_state=2)


gkm = MeanPartitionClustering(n_clusters=nc, n_components=40, n_neighbors=3, trans='spectral', cdistance='ANMI')
res, l = gkm.fit(X, y)


fig = plt.figure()

# ax = fig.gca(projection='3d')
# pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=gkm.labels_, s=25)

ax = fig.add_subplot(111)
plt.scatter(res[:, 0], res[:, 1], c=l)

plt.show()

