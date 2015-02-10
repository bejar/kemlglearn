"""
.. module:: MeanPartition

MeanPartition
*************

:Description: MeanPartition

    

:Authors: bejar
    

:Version: 

:Created on: 10/02/2015 9:40 

"""

__author__ = 'bejar'

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score
from sklearn.manifold import MDS, TSNE, SpectralEmbedding

class MeanPartitionClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    """Consensus Clustering Algorithm based on the estimation of the mean partition
    """
    def __init__(self, n_clusters, base='kmeans', n_components=10):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.cluster_sizes_ = None
        self.base = base
        self.n_components = n_components


    def fit(self, X):
        """
        Clusters the examples
        :param X:
        :return:
        """
        baseclust = []
        if self.base == 'kmeans':
            km = KMeans(n_clusters=self.n_clusters,n_jobs=-1)


        for i in range(self.n_components):
            km.fit(X)
            baseclust.append(km.labels_)

        mdist = np.zeros((self.n_components, self.n_components))

        for i in range(self.n_components-1):
            for j in range(i+1, self.n_components):
                #mdist[i, j] = adjusted_mutual_info_score(baseclust[i], baseclust[j])
                #mdist[i, j] = adjusted_rand_score(baseclust[i], baseclust[j])
                mdist[i, j] = v_measure_score(baseclust[i], baseclust[j])
                mdist[j, i] = mdist[i, j]


        #embed = MDS(dissimilarity='precomputed')
        #embed = TSNE(metric='precomputed')
        embed = SpectralEmbedding(affinity='precomputed', n_neighbors=3)

        X = embed.fit_transform(mdist)

        return X
