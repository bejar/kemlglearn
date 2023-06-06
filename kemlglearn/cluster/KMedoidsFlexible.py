"""
.. module:: KMedoidsFlexible

KMedoidsFlexible
*************

:Description: KMedoidsFlexible


:Authors: bejar

:Version: 

:Created on: 18/12/2017 10:45 

"""
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from numpy.random import choice
import numpy as np

__author__ = 'bejar'

class KMedoidsFlexible(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Medoids algorithm with distance as a parameter

    Partitions a dataset using the K-medoids algorithm with arbitrary distance (passed as a parameter)

    Parameters:

    n_clusters: int
    max_iter: int
    tol: float
    distance: function or "precomputed"

       receives a distance function for the computation able to return array x array distances, it Precomputes the
       distance matrix for faster computation when fit/fit_predict is called, but needs a quadratic amount of memory
       so can not be used for large datasets

       if it is "precomputed" then the data is assumed to be the distance matrix (in condensed form, upper triangular)
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
        if self.distance_ == 'precomputed':
            raise NameError('For precomputed distance use fit_predict')
        else:
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
        if self.distance_ == 'precomputed':
            raise NameError('Cannot predict for new data with precomputed distance')
        else:
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

        if self.distance_ == 'precomputed':
            self.distance_matrix_ = X
            self.nexamp_ = int(np.sqrt(X.shape[0] * 2)) + 1
        else:  # A distance function
            self.nexamp_ = X.shape[0]
            self.distance_matrix_ = np.zeros(self.nexamp_ * (self.nexamp_ - 1) // 2)
            # Condensed distance matrix computation
            for i in range(self.nexamp_):
                for j in range(i+1, self.nexamp_):
                    self.distance_matrix_[self.sel(i, j)] = self.distance_(X[i].reshape(1, -1), X[j].reshape(1, -1))

        # TODO: Implement K-means++ initialization strategy
        # Initializes the medoids randomly
        medoids = np.array(choice(range(self.nexamp_), self.nclusters_, replace=False), dtype=int)
        assignments = np.zeros(self.nexamp_, dtype=int)
        for j in range(self.nexamp_):
            assignments[j] = self._find_nearest_medoid(j, medoids)

        for i in range(self.max_iter_):
            new_medoids = self._kmedoids_iter(assignments)

            if np.array_equal(new_medoids, medoids):
                break
            else:
                medoids = new_medoids
            for j in range(self.nexamp_):
                assignments[j] = self._find_nearest_medoid(j, new_medoids)

        if self.distance_ == 'precomputed':
            # If distance is precomputed we store the indices corresponding to the medoids
            self.cluster_medoids_ = medoids
        else:
            self.cluster_medoids_ = X[medoids, :]

        return assignments

    def _kmedoids_iter(self, assignments):
        """
        One iteration of k medoids
        :param X:
        :param medoids:
        :return:
        """
        lassign = [[] for i in range(self.nclusters_)]
        for i in range(self.nexamp_):
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
        return self.nexamp_ * i - (i * (i+1)//2) + (j - i) - 1

    def sel_distance(self, i, j):
        """
        Selects the distance between two examples from the distance matrix

        :param i: int
        :param j: int
        :return: float
        """
        if i == j:
            return 0
        elif i < j:
            return self.distance_matrix_[self.nexamp_ * i - (i * (i+1)//2) + (j -i) - 1]
        else:
            return self.distance_matrix_[self.nexamp_ * j - (j * (j+1)//2) + (i -j) - 1]


if __name__ == '__main__':
    from kemlglearn.datasets import make_blobs
    from scipy.spatial import distance
    from sklearn.metrics import pairwise
    import matplotlib.pyplot as plt
    from numpy.random import normal

    sc1=75
    v1=0.1
    sc2=75
    v2=0.9
    X = np.zeros((sc1 + sc2, 2))
    X[0:sc1, 0] = normal(loc=-0.5, scale=v1, size=sc1)
    X[0:sc1, 1] = normal(loc=0.0, scale=v2, size=sc1)
    X[sc1:, 0] = normal(loc=0.5, scale=v1, size=sc2)
    X[sc1:, 1] = normal(loc=0.0, scale=v2, size=sc2)
    dlabels = np.zeros(sc1 + sc2)
    dlabels[sc1:] = 1
    cosine = lambda x,y: distance.cdist(x,y, metric='braycurtis')


    # km = KMedoidsFlexible(n_clusters=2, distance=cosine)
    # labels = km.fit_predict(X)
    km = KMedoidsFlexible(n_clusters=2, distance='precomputed')
    D = cosine(X,X)
    D = distance.squareform(D, force='tovector', checks=False)
    labels = km.fit_predict(D)


    fig = plt.figure()

    ax = fig.add_subplot(111)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    medoids = km.cluster_medoids_

    for i, m in enumerate(medoids):
        # plt.scatter(medoids[i, 0], medoids[i, 1], c=i, marker='x', s=200)
        plt.scatter(X[m, 0], X[m, 1], c=i, marker='x', s=200)

    plt.ylim(-3,3)
    plt.xlim(-3,3)
    plt.show()

    print(labels)