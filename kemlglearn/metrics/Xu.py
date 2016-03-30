"""
.. module:: Xu

Xu
*************

:Description: Xu

    

:Authors: bejar
    

:Version: 

:Created on: 22/12/2014 15:26 

"""

__author__ = 'bejar'


import numpy as np
from sklearn.base import BaseEstimator
from kemlglearn.metrics.cluster import within_scatter_matrix_score
from sklearn.cluster import KMeans

class Xu(BaseEstimator):
    """
    Method for determining the number of clusters for numerical data defined in:

    XU, L. (1997) Bayesian Ying-Yang Machine, clustering and Number of Clusters.
    Pattern Recognition Letters 18, 1167-1178

    Implemented using a similar interface to the BaseEstimator class
    """
    def __init__(self, maxc=5, clustering='kmeans'):
        """

        :param maxc: Maximum number of clusters to test (and for computing the fractal dimension
        :param clustering: Clustering algorithm to use
        :return:
        """
        self._maxc = maxc
        self._clustering = clustering
        self._M = None

    def fit(self, X):
        """

        :param X:
        :return:
        """

        lcl = range(1, self._maxc+1)


        # compute the fractal dimension
        ldistorsion = []
        for i in range(1, self._maxc+1):
            cluster = KMeans(n_clusters=i, n_jobs=-1)
            cluster.fit(X)
            ldistorsion.append(within_scatter_matrix_score(X, cluster.labels_))

        print(X.shape[1])
        print(ldistorsion)

        PCF = []
        for x,y in zip(ldistorsion, lcl):
            print(x,y, np.power(y, 2.0/X.shape[1]))
            PCF.append(x * np.power(y, 2.0/X.shape[1]))

        print(PCF)

        self._M = np.argmin(PCF)
        print(self._M)
