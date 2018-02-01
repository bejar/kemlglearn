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
from numpy.random import randint

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
    def __init__(self, n_clusters, n_clusters_base=None, ncb_rand=False, base='kmeans', n_components=10,
                 consensus='coincidence', consensus2='kmeans'):
        self.n_clusters = n_clusters
        if n_clusters_base is None:
            self.n_clusters_base = n_clusters
        else:
            self.n_clusters_base = n_clusters_base
        self.ncb_rand = ncb_rand
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

        for i in range(self.n_components):
            ncl = self.n_clusters_base if self.ncb_rand else randint(2, self.n_clusters_base+1)
            if self.base == 'kmeans':
                km = KMeans(n_clusters=ncl, n_init=1, init='random')
            elif self.base == 'spectral':
                km = SpectralClustering(n_clusters=ncl, assign_labels='discretize',
                                        affinity='nearest_neighbors', n_neighbors=30)
            km.fit(X)
            baseclust.append(km.labels_)

        coin_matrix = np.zeros((X.shape[0], X.shape[0]))

        for l in baseclust:
            for i in range(X.shape[0]):
                coin_matrix[i, i] += 1
                for j in range(i+1, X.shape[0]):
                    #if i != j:
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


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.metrics import adjusted_mutual_info_score
    from kemlglearn.datasets import make_blobs
    import matplotlib.pyplot as plt


    # data = datasets.load_iris()['data']
    # labels = datasets.load_iris()['target']

    # data, labels = make_blobs(n_samples=[100, 200], n_features=2, centers=[[1,1], [0,0]], random_state=2, cluster_std=[0.2, 0.4])
    data, labels = datasets.make_circles(n_samples=400, noise=0.1, random_state=4, factor=0.3)

    km = KMeans(n_clusters=2)

    cons = SimpleConsensusClustering(n_clusters=2, n_clusters_base=20, n_components=50, ncb_rand=False)

    lkm = km.fit_predict(data)
    cons.fit(data)
    lcons = cons.labels_

    print(adjusted_mutual_info_score(lkm, labels))
    print(adjusted_mutual_info_score(lcons, labels))

    fig = plt.figure()

    # ax = fig.gca(projection='3d')
    # pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=ld.labels_, s=25)
    #
    ax = fig.add_subplot(131)
    plt.scatter(data[:,0],data[:,1],c=labels)
    ax = fig.add_subplot(132)
    plt.scatter(data[:,0],data[:,1],c=lkm)
    ax = fig.add_subplot(133)
    plt.scatter(data[:,0],data[:,1],c=lcons)

    plt.show()
