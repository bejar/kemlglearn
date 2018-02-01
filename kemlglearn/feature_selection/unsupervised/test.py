"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 25/11/2014 11:15 

"""

__author__ = 'bejar'

from amltlearn.feature_selection.unsupervised import LaplacianScore

from sklearn.datasets import make_blobs
import pylab as pl
import matplotlib.pyplot as plt


X,y_data = make_blobs(n_samples=5200, n_features=20, centers=7, random_state=0)

LS = LaplacianScore(n_neighbors=10, bandwidth=0.05)

LS.fit(X)

print(LS.scores_)
best = LS.best_k_scores(2)
print(best)

fig = plt.figure()

# ax = fig.gca(projection='3d')
# pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=ld.labels_, s=25)
#
ax = fig.add_subplot(111)
plt.scatter(X[:,best[0]],X[:,best[1]],c=y_data)

plt.show()

