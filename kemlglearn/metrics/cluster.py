"""
.. module:: cluster

cluster
*************

:Description: cluster

 Measures for clustering quality


:Authors: bejar
    

:Version: 

:Created on: 22/12/2014 13:21 

"""

__author__ = 'bejar'


import numpy as np
from  sklearn.metrics.pairwise import euclidean_distances

def within_scatter_matrix_score(X, labels):
    """
    Computes the within scatter matrix score (the distorsion) of a labeling of a clustering
    :param X:
    :param labels:
    :return:
    """

    llabels = np.unique(labels)
    print llabels

    dist = 0.0
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += X[center_mask].sum()
        center /= center_mask.sum()
        dvector = euclidean_distances(X[center_mask], center)
        dist += dvector.sum()
    return dist / len(labels)



