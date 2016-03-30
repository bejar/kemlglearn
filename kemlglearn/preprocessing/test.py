"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 13/03/2015 16:34 

"""

__author__ = 'bejar'

from kemlglearn.preprocessing import Discretizer

from sklearn.datasets import make_blobs, load_iris, make_circles

X = load_iris()['data']

print(X.shape)

disc = Discretizer(bins=3, method='frequency')

disc.fit(X)

print(disc.intervals)

disc = Discretizer(bins=3, method='equal')

disc.fit(X)

print(disc.intervals)


# Y = disc.transform(X, copy=True)
#
#
# for i in range(X.shape[0]):
#     print X[i,:], Y[i,:]
