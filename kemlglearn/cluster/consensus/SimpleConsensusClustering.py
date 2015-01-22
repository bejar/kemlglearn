"""
.. module:: SimpleConsensusClustering

SimpleConsensusClustering
*************

:Description: SimpleConsensusClustering

    

:Authors: bejar
    

:Version: 

:Created on: 22/01/2015 10:46 

"""

__author__ = 'bejar'


import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


class SimpleConsensusClustering(BaseEstimator,ClusterMixin,TransformerMixin):
    """Simple Consensus Clustering Algorithm

    Paramerets:

    radius: float
        Clustering radius for asigning examples to a cluster

    """
    def __init__(self, n_clusters, base='kmeans', n_components=10, consensus='coincidence'):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.cluster_sizes_ = None
        self.base = base
        self.n_components = n_components
        self.consensus = consensus


    def fit(self,X):
        """
        Clusters the examples
        :param X:
        :return:
        """

        if self.base == 'kmeans':
            self.cluster_centers_, self.labels_ = self._fit_process_kmeans(X)


    def _fit_process_kmeans(self,X):
        """
        Obtains n_components kmeans clustering, compute the coincidence matrix and applies kmeans to that coincidence
        matrix

        :param X:
        :return:
        """
        baseclust = []
        km = KMeans(n_clusters=self.n_clusters,n_jobs=-1)
        for i in range(self.n_components):
            km.fit(X)
            baseclust.append(km.labels_)

        coin_matrix = np.zeros((X.shape[0], X.shape[0]))

        for l in baseclust:
            for i in range(X.shape[0]):
                for j in range(i+1, X.shape[0]):
                    if i != j:
                        if l[i] == l[j]:
                            coin_matrix[i, j] += 1
                            coin_matrix[j, i] += 1


        coin_matrix /= self.n_components
