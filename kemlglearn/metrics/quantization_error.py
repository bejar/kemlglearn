"""
.. module:: cluster

cluster
*************

:Description: cluster

    

:Authors: bejar
    

:Version: 

:Created on: 22/12/2014 13:09 

"""

__author__ = 'bejar'

import numpy as np
from sklearn.base import BaseEstimator
from kemlglearn.metrics.cluster import within_scatter_matrix_score
from sklearn.cluster import KMeans

class quantization_error(BaseEstimator):
    """
    Method for determining the number of clusers for numerical data defined in:

    Kolesnikov, A.; Trichina, E. & Kauranne, T. (2015) "Estimating the number of clusters in a numerical data set via
    quantization error modeling" Pattern Recognition ,48, 941 - 952

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
        self._fdim = None
        self._M = None

    def fit(self, X):
        """

        :param X:
        :return:
        """

        lcl = range(1, self._maxc+1)

        logsum = np.log(lcl).sum()
        num = self._maxc * (np.log(lcl)*np.log(lcl)).sum()
        num -= logsum*logsum

        print(num)
        # compute the fractal dimension
        ldistorsion = []
        for i in range(1, self._maxc+1):
            cluster = KMeans(n_clusters=i, n_jobs=-1)
            cluster.fit(X)
            ldistorsion.append(cluster.inertia_/X.shape[0])

        print(ldistorsion)
        den = self._maxc * (np.log(lcl) * np.log(ldistorsion)).sum()
        den -= (logsum * np.log(ldistorsion)).sum()

        print(den)
        self._fdim = 2 * num / den

        print(self._fdim)
        PCF = [x * np.power(y, 2/self._fdim) for x, y in zip(ldistorsion, lcl)]

        print(PCF)

        self._M = np.argmin(PCF)
        print(self._M)


