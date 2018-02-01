"""
.. module:: GlobalKMeans

GlobalKMeans
*************

:Description: GlobalKMeans

    

:Authors: bejar
    

:Version: 

:Created on: 20/01/2015 10:42 

"""

__author__ = 'bejar'

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class GlobalKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Global K-means Algorithm

    Paramereters:

    n_clusters: int
        maximum number of clusters to obtain
    algorithm string
        'classical' the classical algorithm
        'bagirov' the Bagirov 2006 variant

    """

    def __init__(self, n_clusters, algorithm='classical'):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.cluster_sizes_ = None
        self.inertia_ = None
        self.algorithm = algorithm

    def fit(self, X):
        """
        Clusters the examples
        :param X:
        :return:
        """

        if self.algorithm == 'classical':
            self.cluster_centers_, self.labels_, self.inertia_ = self._fit_process(X)
        elif self.algorithm == 'bagirov':
            self.cluster_centers_, self.labels_, self.inertia_ = self._fit_process_bagirov(X)

        return self

    def predict(self, X):
        """
        Returns the nearest cluster for a data matrix

        @param X:
        @return:
        """
        clasif = []
        for i in range(X.shape[0]):
            ncl, mdist = self._find_nearest_cluster(X[i].reshape(1, -1), self.cluster_centers_)
            if mdist <= self.radius:
                clasif.append(ncl)
            else:
                clasif.append(-1)
        return clasif

    def _fit_process(self, X):
        """
        Classical global k-means algorithm

        :param X:
        :return:
        """

        # Compute the centroid of the dataset
        centroids = sum(X) / X.shape[0]
        centroids.shape = (1, X.shape[1])

        for i in range(2, self.n_clusters + 1):
            mininertia = np.infty
            for j in range(X.shape[0]):
                newcentroids = np.vstack((centroids, X[j]))
                # print newcentroids.shape
                km = KMeans(n_clusters=i, init=newcentroids, n_init=1)
                km.fit(X)
                if mininertia > km.inertia_:
                    mininertia = km.inertia_
                    bestkm = km
            centroids = bestkm.cluster_centers_

        return bestkm.cluster_centers_, bestkm.labels_, bestkm.inertia_

    def _fit_process_bagirov(self, X):
        """
        Clusters using the global K-means algorithm Bagirov variation
        :param X:
        :return:
        """

        # Create a KNN structure for fast search
        self._neighbors = NearestNeighbors()
        self._neighbors.fit(X)

        # Compute the centroid of the dataset
        centroids = sum(X) / X.shape[0]
        assignments = [0 for i in range(X.shape[0])]

        centroids.shape = (1, X.shape[1])

        # compute the distance of the examples to the centroids
        mindist = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            mindist[i] = \
            euclidean_distances(X[i].reshape(1, -1), centroids[assignments[i]].reshape(1, -1), squared=True)[0]

        for k in range(2, self.n_clusters + 1):
            newCentroid = self._compute_next_centroid(X, centroids, assignments, mindist)
            centroids = np.vstack((centroids, newCentroid))
            km = KMeans(n_clusters=k, init=centroids, n_init=1)
            km.fit(X)
            assignments = km.labels_
            for i in range(X.shape[0]):
                mindist[i] = \
                euclidean_distances(X[i].reshape(1, -1), centroids[assignments[i]].reshape(1, -1), squared=True)[0]

        return km.cluster_centers_, km.labels_, km.inertia_

    def _compute_next_centroid(self, X, centroids, assignments, mindist):
        """
        Computes the candidate for the next centroid

        :param X:
        :param centroids:
        :return:
        """
        minsum = np.infty
        candCentroid = None

        # Compute the first candidate to new centroid
        for i in range(X.shape[0]):
            distance = euclidean_distances(X[i].reshape(1, -1), centroids[assignments[i]].reshape(1, -1))[0]
            S2 = self._neighbors.radius_neighbors(X[i].reshape(1, -1), radius=distance, return_distance=False)[0]
            S2centroid = np.sum(X[S2], axis=0) / len(S2)
            S2centroid.shape = (1, X.shape[1])
            cost = self._compute_fk(X, mindist, S2centroid)

            if cost < minsum:
                minsum = cost
                candCentroid = S2centroid

        # Compute examples for the new centroid
        S2 = []
        newDist = euclidean_distances(X, candCentroid.reshape(1, -1), squared=True)
        for i in range(X.shape[0]):
            if newDist[i] < mindist[i]:
                S2.append(i)

        newCentroid = sum(X[S2]) / len(S2)
        newCentroid.shape = (1, X.shape[1])

        while not (candCentroid == newCentroid).all():
            candCentroid = newCentroid
            S2 = []
            newDist = euclidean_distances(X, candCentroid.reshape(1, -1), squared=True)
            for i in range(X.shape[0]):
                if newDist[i] < mindist[i]:
                    S2.append(i)

            newCentroid = np.sum(X[S2], axis=0) / len(S2)
            newCentroid.shape = (1, X.shape[1])

        return candCentroid

    def _compute_fk(self, X, mindist, ccentroid):
        """
        Computes the cost function

        :param X:
        :param mindist:
        :param ccentroid:
        :return:
        """

        # Distances among the examples and the candidate centroid
        centdist = euclidean_distances(X, ccentroid.reshape(1, -1), squared=True)

        fk = 0
        for i in range(X.shape[0]):
            fk = fk + min(mindist[i], centdist[i][0])

        return fk

    @staticmethod
    def _find_nearest_cluster(examp, centers):
        """
        Finds the nearest cluster for an example
        :param examp:
        :param centers:
        :return:
        """

        dist = euclidean_distances(centers, examp.reshape(1, -1))

        pmin = np.argmin(dist)
        vmin = np.min(dist)

        return pmin, vmin
