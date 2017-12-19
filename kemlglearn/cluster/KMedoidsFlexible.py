"""
.. module:: KMedoidsFlexible

KMedoidsFlexible
*************

:Description: KMedoidsFlexible

    

:Authors: bejar
    

:Version: 

:Created on: 18/12/2017 10:45 

"""

from __future__ import division

__author__ = 'bejar'


import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from numpy.random import choice

class KMedoidsFlexible(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Medoids algorithm with distance as a parameter

    Partitions a dataset using the K-medoids algorithm with arbitrary distance (passed as a parameter)

    Parameters:

    n_clusters: int
    max_iter: int
    tol: float
    distance: distance function for the computation
    Precomputes the distance for faster computation, but needs a quadratic amount of memory so can not be used for large
    datasets
    """

    def __init__(self, n_clusters=3, max_iter=50, distance=euclidean_distances, random_state=None):
        self.cluster_medoids_ = None
        self.labels_ = None
        self.distance_ = distance
        self.nclusters_ = n_clusters
        self.max_iter_ = max_iter
        self.distance_matrix_ = None
        self.random_ = random_state

    def fit(self, X):
        """
        Clusters the examples
        :param X:
        :return:
        """
        self._fit_process(X)

        return self

    def fit_predict(self, X):
        """
        Clusters the examples
        :param X:
        :return:
        """
        self.labels_ = self._fit_process(X)

        return self.labels_


    def predict(self, X):
        """
        Returns the nearest cluster for a data matrix

        @param X:
        @return:
        """
        clasif = []
        for i in range(X.shape[0]):
            clasif.append(np.argmin(self.distance_(X[i].reshape(1, -1), self.cluster_medoids_)))
        return clasif

    def _fit_process(self, X):
        """
        Computes K-medoids
        :param X:
        :return:
        """

        self.distance_matrix_ = np.zeros(X.shape[0] * (X.shape[0] - 1) // 2)
        self.nexamp_ = X.shape[0]
        for i in range(X.shape[0]):
            for j in range(i+1, X.shape[0]):
                self.distance_matrix_[self.sel(i, j)] = self.distance_(X[i].reshape(1, -1), X[j].reshape(1, -1))

        # Initializes the medoids using k-means
        km = KMeans(n_clusters=self.nclusters_)
        assignments = km.fit_predict(X)
        medoids = np.zeros(self.nclusters_, dtype=int)-1
        for i in range(self.max_iter_):
            new_medoids = self._kmedoids_iter(X, assignments)
            print(medoids, new_medoids)
            if np.array_equal(new_medoids, medoids):
                break
            else:
                medoids = new_medoids

            for i in range(X.shape[0]):
                assignments[i] = self._find_nearest_medoid(i, medoids)
            print(medoids)

        self.cluster_medoids_ = X[medoids, :]

        return assignments

    def _kmedoids_iter(self, X, assignments):
        """
        One iteration of k medoids
        :param X:
        :param medoids:
        :return:
        """
        lassign = [[] for i in range(self.nclusters_)]
        for i in range(X.shape[0]):
            lassign[assignments[i]].append(i)

        # Compute the new medoids
        new_medoids = np.zeros(self.nclusters_, dtype=int)
        for i, ass in enumerate(lassign):
            mndist = np.inf
            for a1 in ass:
                mddist = np.sum([self.sel_distance(a1, c) for c in ass])
                if mddist < mndist:
                    new_medoids[i] = a1
                    mndist = mddist

        return new_medoids

    def _find_nearest_medoid(self, examp, centers):
        """
        Finds the nearest cluster for an example
        :param examp:
        :param centers:
        :return:
        """
        return np.argmin([self.sel_distance(examp, c) for c in centers])

    def sel(self, i, j):
        return self.nexamp_ * i - (i * (i+1)//2) + (j -i) - 1

    def sel_distance(self, i, j):
        """
        Selects the distance between two examples from the distance matrix

        Prec: i < j

        :param d:
        :param i:
        :param j:
        :return:
        """
        if i == j:
            return 0
        elif i < j:
            return self.distance_matrix_[self.nexamp_ * i - (i * (i+1)//2) + (j -i) - 1]
        else:
            return self.distance_matrix_[self.nexamp_ * j - (j * (j+1)//2) + (i -j) - 1]
