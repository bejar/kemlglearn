"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 07/07/2014 11:12 

"""
from Leader import Leader

__author__ = 'bejar'

from sklearn.datasets import make_blobs
import pylab as pl
import matplotlib.pyplot as plt


X,y_data = make_blobs(n_samples=5200, n_features=20, centers=7, random_state=0)

ld = Leader(radius=25.0)

#print timeit.timeit(stmt='ld.fit(X)',setup=setup,number=10)
ld.fit(X)

# print ld.cluster_centers_.shape[0]

fig = plt.figure()

# ax = fig.gca(projection='3d')
# pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=ld.labels_, s=25)
#
ax = fig.add_subplot(111)
plt.scatter(X[:,0],X[:,1],c=ld.labels_)

plt.show()

