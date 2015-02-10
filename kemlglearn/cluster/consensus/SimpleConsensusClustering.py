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
from sklearn.cluster import KMeans, SpectralClustering


class SimpleConsensusClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    """Simple Consensus Clustering Algorithm

    Pararemeters:

    n_clusters: int
        Number of clusters of the base clusterers and the consensus cluster
    base: string
        base clusterer ['kmeans']
    n_components: int
        number of components of the consensus
    consensus: string
        consensus method ['coincidence']

    """
    def __init__(self, n_clusters, base='kmeans', n_components=10,
                 consensus='coincidence', consensus2='kmeans'):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.cluster_sizes_ = None
        self.base = base
        self.n_components = n_components
        self.consensus = consensus
        self.consensus2 = consensus2


    def fit(self, X):
        """
        Clusters the examples
        :param X:
        :return:
        """

        if self.consensus == 'coincidence':
            self.cluster_centers_, self.labels_ = self._fit_process_coincidence(X)


    def _fit_process_coincidence(self, X):
        """
        Obtains n_components kmeans clustering, compute the coincidence matrix and applies kmeans to that coincidence
        matrix

        :param X:
        :return:
        """
        baseclust = []
        if self.base == 'kmeans':
            km = KMeans(n_clusters=self.n_clusters, n_jobs=-1)
        elif self.base == 'spectral':
            km = SpectralClustering(n_clusters=self.n_clusters, assign_labels='discretize',
                                    affinity='nearest_neighbors', n_neighbors=30)

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
        if self.consensus2 == 'kmeans':
            kmc = KMeans(n_clusters=self.n_clusters, n_jobs=-1)
            kmc.fit(coin_matrix)
            return kmc.cluster_centers_, kmc.labels_
        elif self.consensus2 == 'spectral':
            kmc = SpectralClustering(n_clusters=self.n_clusters, assign_labels='discretize',
                                    affinity='nearest_neighbors', n_neighbors=40)
            kmc.fit(coin_matrix)

            return None, kmc.labels_


