"""
.. module:: Leader

Leader
*************

:Description: Leader Algorithm Clustering

    

:Authors: bejar
    

:Version: 

:Created on: 07/07/2014 8:29 

"""

__author__ = 'bejar'

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances


class Leader(BaseEstimator, ClusterMixin, TransformerMixin):
    """Leader Algorithm Clustering

    Partition of a dataset using the leader algorithm

    Parameters:

    radius: float
        Clustering radius for asigning examples to a cluster

    """

    def __init__(self, radius):
        self.radius = radius
        self.cluster_centers_ = None
        self.labels_ = None
        self.cluster_sizes_ = None

    def num_clusters(self):
        return self.cluster_centers_.shape[0]

    def fit(self, X):
        """
        Clusters the examples
        :param X:
        :return:
        """

        self.cluster_centers_, self.labels_, self.cluster_sizes_ = self._fit_process(X)

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
        Clusters incrementally the examples
        :param X:
        :return:
        """
        assignments = []
        scenters = np.zeros((1, X.shape[1]))
        centers = np.zeros((1, X.shape[1]))
        # Initialize with the first example
        scenters[0] = X[0]
        centers[0] = X[0]
        assignments.append([0])
        csizes = np.array([1])
        # Cluster the rest of examples
        for i in range(1, X.shape[0]):
            ncl, mdist = self._find_nearest_cluster(X[i].reshape(1, -1), centers)

            # if distance is less than radius, introduce example in nearest class
            if mdist <= self.radius:
                scenters[ncl] += X[i]
                csizes[ncl] += 1
                centers[ncl] = scenters[ncl] / csizes[ncl]
                assignments[ncl].append(i)
            else:  # Create a new cluster
                scenters = np.append(scenters, np.array([X[i]]), 0)
                centers = np.append(centers, np.array([X[i]]), 0)
                csizes = np.append(csizes, [1], 0)
                assignments.append([i])

        labels = np.zeros(X.shape[0])
        for l, ej in enumerate(assignments):
            for e in ej:
                labels[e] = l

        return centers, labels, csizes

    @staticmethod
    def _find_nearest_cluster(examp, centers):
        """
        Finds the nearest cluster for an example
        :param examp:
        :param centers:
        :return:
        """
        dist = euclidean_distances(centers, examp)

        pmin = np.argmin(dist)
        vmin = np.min(dist)

        return pmin, vmin


if __name__ == '__main__':
    from sklearn.datasets import make_blobs, load_iris, make_circles

    X, y_data = make_circles(n_samples=1000, noise=0.5, random_state=4, factor=0.5)
    ld = Leader(radius=.01)
    ld.fit(X)
    print(ld.predict(np.array([[0, 0]])))
