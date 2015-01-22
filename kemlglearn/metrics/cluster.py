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
from sklearn.metrics.pairwise import euclidean_distances

def scatter_matrices_scores(X, labels, indices=['CH']):
    """
    Computes differente indices obtained from the Within and Between scatter matrices

    Includes:
        'SSW': Within scatter matrix score
        'SSB': Between scatter matrix score
        'Hartigan': Hartigan index
        'CH': Caliski-Harabasz index
        'Xu': Xu index
        'ZCF': ZhaoChuFranti index

    :param X:
    :param labels:
    :return:
    """
    llabels = np.unique(labels)
    nclust = len(llabels)
    nex = len(labels)

    # Centroid of the data

    centroid = np.zeros((1, X.shape[1]))
    centroid += np.sum(X, axis=0)
    centroid /= X.shape[0]

    # Compute SSB
    ccentroid = np.zeros((nclust, X.shape[1]))
    dist = 0.0
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        ccentroid[idx] = center
        dvector = euclidean_distances(centroid, ccentroid[idx], squared=True)
        dist += dvector.sum() * center_mask.sum()
    SSB = dist / len(labels)

    # Compute SSW
    dist = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], ccentroid[idx], squared=True)
        dist += dvector.sum()

    SSW = dist / len(labels)

    results = {}

    if 'CH' in indices:
        results['CH'] = (SSB/(nclust-1))/(SSW/(nex-nclust))

    if 'Hartigan' in indices:
        results['Hartigan'] = -np.log(SSW/SSB)

    if 'ZCF' in indices:
        results['ZCF'] = (SSW/SSB) * nclust

    if 'Xu' in indices:
        results['Xu'] = X.shape[1] * np.log(np.sqrt(SSW/(X.shape[1]*nex*nex)))+np.log(nclust)

    if 'SSW' in indices:
        results['SSW'] = SSW

    if 'SSB' in indices:
        results['SSB'] = SSB

    return results


def within_scatter_matrix_score(X, labels):
    """
    Computes the within scatter matrix score of a labeling of a clustering
    :param X:
    :param labels:
    :return:
    """

    llabels = np.unique(labels)

    dist = 0.0
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        dvector = euclidean_distances(X[center_mask], center, squared=True)
        dist += dvector.sum()
    return dist / len(labels)


def between_scatter_matrix_score(X, labels):
    """
    Computes the between scatter matrix score of a labeling of a clustering
    :param X:
    :param labels:
    :return:
    """

    llabels = np.unique(labels)

    # Centroid of the data

    centroid = np.zeros((1, X.shape[1]))
    centroid += np.sum(X, axis=0)
    centroid /= X.shape[0]

    dist = 0.0
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        dvector = euclidean_distances(centroid, center, squared=True)
        dist += dvector.sum() * center_mask.sum()
    return dist / len(labels)


def CalinskiHarabasz(X, labels):
    """
    Computes the Calinski&Harabasz score for a labeling of the data

    :param X:
    :param labels:
    :return:
    """
    llabels = np.unique(labels)

    # Centroid of the data

    centroid = np.zeros((1, X.shape[1]))
    centroid += np.sum(X, axis=0)
    centroid /= X.shape[0]

    # Compute SSB
    ccentroid = np.zeros((len(llabels), X.shape[1]))
    dist = 0.0
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        ccentroid[idx] = center
        dvector = euclidean_distances(centroid, ccentroid[idx], squared=True)
        dist += dvector.sum() * center_mask.sum()
    SSB = dist / len(labels)

    # Compute SSW
    dist = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], ccentroid[idx], squared=True)
        dist += dvector.sum()

    SSW = dist / len(labels)

    return (SSB/(len(llabels)-1))/(SSW/(len(labels)-len(llabels)))


def ZhaoChuFranti(X, labels):
    """
    Implements the method defined in:

    Zhao, Q.; Xu, M. & Franti, P.  Sum-of-Squares Based Cluster Validity Index and Significance Analysis
    Adaptive and Natural Computing Algorithms, Springer Berlin Heidelberg, 2009, 5495, 313-322

    :param X:
    :param labels:
    :return:
    """
    llabels = np.unique(labels)

    # Centroid of the data

    centroid = np.zeros((1, X.shape[1]))
    centroid += np.sum(X, axis=0)
    centroid /= X.shape[0]

    # Compute SSB
    ccentroid = np.zeros((len(llabels), X.shape[1]))
    dist = 0.0
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        ccentroid[idx] = center
        dvector = euclidean_distances(centroid, ccentroid[idx], squared=True)
        dist += dvector.sum() * center_mask.sum()
    SSB = dist / len(labels)

    # Compute SSW
    dist = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], ccentroid[idx], squared=True)
        dist += dvector.sum()

    SSW = dist / len(labels)

    return (SSW/SSB) * len(llabels)


def DaviesBouldin(X, labels):
    """
    Implements the Davies&Bouldin score for a labeling of the data

    :param X:
    :param labels:
    :return:
    """

    llabels = np.unique(labels)
    nclust = len(llabels)

    # compute the centroids
    centroids = np.zeros((nclust, X.shape[1]))
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        centroids[idx] = center

    # Centroids distance matrix
    cdistances = euclidean_distances(centroids)

    # Examples to centroid mean distance
    mdcentroid = np.zeros(nclust)
    for idx in llabels:
        center_mask = labels == idx
        vdist = euclidean_distances(centroids[idx], X[center_mask])
        mdcentroid[idx] = vdist.sum()/center_mask.sum()

    # Compute the index
    dist = 0.0
    for idxi in llabels:
        lvals = []
        disti = mdcentroid[idxi]
        for idxj in llabels:
            if idxj != idxi:
                lvals.append((disti + mdcentroid[idxj])/cdistances[idxi, idxj])
        dist += max(lvals)

    return dist/nclust
