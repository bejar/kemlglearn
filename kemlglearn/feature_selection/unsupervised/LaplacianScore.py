"""
.. module:: LaplacianScore

LaplacianScore
*************

:Description: LaplacianScore

    Class that computes the laplacian score for a dataset

:Authors: bejar
    

:Version: 

:Created on: 25/11/2014 9:32 

"""

__author__ = 'bejar'

import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from operator import itemgetter

class LaplacianScore():
    """
    Laplacian Score algorithm

    Parameters

        - Number of neighbors to compute the similarity matrix
        - Bandwidth for the gaussian similarity kernel
    """

    scores_ = None


    def __init__(self, n_neighbors=5, bandwidth=0.01, k=None):
        """
        Initial values of the parameters
        @param n_neighbors: Number of neighbors for the spectral matrix
        @param bandwidth: Bandwidth for the gaussian kernel
        @param k: number of features to select
        @return:
        """
        self._n_neighbors = n_neighbors
        self._bandwidth = bandwidth
        self._k = k

    def fit(self, X):
        """
        Computes the laplacian scores for the dataset

        X is a [n_examples, n_attributes] numpy array
        @return:
        """

        self._fit_process(X)

        return self

    def _best_k_scores(self, k=5):
        """
        returns the indices of the best k attributes according to the score

        :param k:
        :return:
        """
        if self.scores_ is None:
            raise Exception('Laplacian Score: Not fitted')
        else:
            l = list(enumerate(self.scores_))
            l = sorted(l, key=itemgetter(1), reverse=True)
            return [l[i][0] for i in range(k)]

    def fit_transform(self, X):
        """
        Selects the features and returns the dataset with only the k best ones
        :return:
        """

        self._fit_process(X)
        l = list(enumerate(self.scores_))
        l = sorted(l, key=itemgetter(1), reverse=True)
        lsel = [l[i][0] for i in range(self._k)]
        return X[:, lsel]

    # Todo: implementation only with sparse matrices
    def _fit_process(self, X):
        """
        Computes the Laplacian score for the attributes


        @param X:
        @return:
        """

        self.scores_ = np.zeros(X.shape[1])

        # Similarity matrix
        S = kneighbors_graph(X, n_neighbors=self._n_neighbors, mode='distance')
        S = S.toarray()
        S *= S
        S /= self._bandwidth
        S = -S

        ones = np.ones(X.shape[0])

        D = np.diag(np.dot(S, ones))

        L = D - S

        qt = D.sum()
        for at in range(X.shape[1]):
            Fr = X[:, at]
            Fr_hat = Fr - np.dot(np.dot(Fr, D) / qt, ones)

            score1 = np.dot(np.dot(Fr_hat, L), Fr_hat)
            score2 = np.dot(np.dot(Fr_hat, D), Fr_hat)
            self.scores_[at] = score1 / score2




