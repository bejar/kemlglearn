"""
.. module:: cluster

cluster
*************

:Description: cluster

 Measures for clustering quality


:Authors: bejar
    

:Version: 

:Created on: 22/12/2014 13:21

# 22/09/2015: Fix: The labels from a clusterer from scikit-learn are not always contiguous

"""

__author__ = 'bejar'


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster.supervised import contingency_matrix, check_clusterings, mutual_info_score, entropy

def maplabels(labels):
    """
    Returns a dictionary mapping a set of labels to an index

    :param labels:
    :return:
    """
    poslabels = {}
    for lab, p in zip(labels,range(len(labels))):
        poslabels[lab] = p
    return poslabels

def scatter_matrices_scores(X, labels, indices=['CH']):
    """
    Computes different indices obtained from the Within and Between scatter matrices

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
    poslabels = maplabels(llabels)

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
        ccentroid[poslabels[idx]] = center
        dvector = euclidean_distances(centroid.reshape(1, -1), ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
        dist += dvector.sum() * center_mask.sum()
    SSB = dist / len(labels)

    # Compute SSW
    dist = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
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

    if 'Inertia' in indices:
        results['Inertia'] = SSW * len(labels)

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
        dvector = euclidean_distances(X[center_mask].reshape(1, -1), center, squared=True)
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
        dvector = euclidean_distances(centroid.reshape(1, -1), center.reshape(1, -1), squared=True)
        dist += dvector.sum() * center_mask.sum()
    return dist / len(labels)


def calinski_harabasz_score(X, labels):
    """
    Computes the Calinski&Harabasz score for a labeling of the data

    :param X:
    :param labels:
    :return:
    """
    llabels = np.unique(labels)
    poslabels = maplabels(llabels)

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
        ccentroid[poslabels[idx]] = center
        dvector = euclidean_distances(centroid.reshape(1, -1), ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
        dist += dvector.sum() * center_mask.sum()
    SSB = dist / len(labels)

    # Compute SSW
    dist = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
        dist += dvector.sum()

    SSW = dist / len(labels)

    return (SSB/(len(llabels)-1))/(SSW/(len(labels)-len(llabels)))


def zhao_chu_franti_score(X, labels):
    """
    Implements the method defined in:

    Zhao, Q.; Xu, M. & Franti, P.  Sum-of-Squares Based Cluster Validity Index and Significance Analysis
    Adaptive and Natural Computing Algorithms, Springer Berlin Heidelberg, 2009, 5495, 313-322

    :param X:
    :param labels:
    :return:
    """
    llabels = np.unique(labels)
    poslabels = maplabels(llabels)

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
        ccentroid[poslabels[idx]] = center
        dvector = euclidean_distances(centroid.reshape(1, -1), ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
        dist += dvector.sum() * center_mask.sum()
    SSB = dist / len(labels)

    # Compute SSW
    dist = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
        dist += dvector.sum()

    SSW = dist / len(labels)

    return (SSW/SSB) * len(llabels)


def davies_bouldin_score(X, labels):
    """
    Implements the Davies&Bouldin score for a labeling of the data

    :param X:
    :param labels:
    :return:
    """

    llabels = np.unique(labels)
    poslabels = maplabels(llabels)
    nclust = len(llabels)

    # compute the centroids
    centroids = np.zeros((nclust, X.shape[1]))
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        centroids[poslabels[idx]] = center

    # Centroids distance matrix
    cdistances = euclidean_distances(centroids)

    # Examples to centroid mean distance
    mdcentroid = np.zeros(nclust)
    for idx in llabels:
        center_mask = labels == idx
        vdist = euclidean_distances(centroids[poslabels[idx]].reshape(1, -1), X[center_mask])
        mdcentroid[poslabels[idx]] = vdist.sum()/center_mask.sum()

    # Compute the index
    dist = 0.0
    for idxi in llabels:
        lvals = []
        disti = mdcentroid[poslabels[idxi]]
        for idxj in llabels:
            if idxj != idxi:
                lvals.append((disti + mdcentroid[poslabels[idxj]])/cdistances[poslabels[idxi], poslabels[idxj]])
        dist += max(lvals)

    return dist/nclust


def jeffrey_divergence_score(X, labels):
    """
    Implements the score based on the Jeffrey divergence that appears in:

    Said, A.; Hadjidj, R. & Foufou, S. "Cluster validity index based on Jeffrey divergence"
    Pattern Analysis and Applications, Springer London, 2015, 1-11

    :param X:
    :param labels:
    :return:
    """

    llabels = np.unique(labels)
    poslabels = maplabels(llabels)
    nclust = len(llabels)

    # compute the centroids
    centroids = np.zeros((nclust, X.shape[1]))
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        centroids[poslabels[idx]] = center

    lcovs = []
    linvcovs = []
    for idx in llabels:
        cov_mask = labels == idx
        covar = np.cov(X[cov_mask].T)
        lcovs.append(covar)
        linvcovs.append(np.linalg.inv(covar))

    traces = np.zeros((nclust, nclust))
    for idx1 in llabels:
        for idx2 in llabels:
            traces[poslabels[idx1], poslabels[idx2]] = np.trace(np.dot(linvcovs[poslabels[idx1]], lcovs[poslabels[idx2]]))
            traces[poslabels[idx1], poslabels[idx2]] += np.trace(np.dot(linvcovs[poslabels[idx2]], lcovs[poslabels[idx1]]))
            traces[poslabels[idx1], poslabels[idx2]] /= 2.0

    sumcov = np.zeros((nclust, nclust))
    for idx1 in llabels:
        for idx2 in llabels:
            v1 = centroids[poslabels[idx1]]
            v2 = centroids[poslabels[idx2]]
            vm = v1-v2
            mcv = linvcovs[poslabels[idx1]] + linvcovs[poslabels[idx2]]
            sumcov[poslabels[idx1], poslabels[idx2]] = np.dot(vm.T, np.dot(mcv, vm))
            sumcov[poslabels[idx1], poslabels[idx2]] /= 2.0

    ssep = 0.0
    for idx1 in llabels:
        minv = np.inf
        for idx2 in llabels:
            if idx1 != idx2:
                val = traces[poslabels[idx1], poslabels[idx2]] + sumcov[poslabels[idx1], poslabels[idx2]] - centroids.shape[1]
                if minv > val:
                    minv = val
        ssep += minv

    scompact = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], centroids[poslabels[idx]], squared=True)
        scompact += dvector.max()

    return  scompact/ssep

def variation_of_information_score(labels_true, labels_pred):
    """Variation of Information (Meila, 2003)
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    mutual = mutual_info_score(labels_true, labels_pred)
    e1 = entropy(labels_true)
    e2 = entropy(labels_pred)

    return e1 + e2 - (2* mutual)


def jaccard_score(labels_true, labels_pred):
    """
    Jaccard coeficient computed according to:

    Ceccarelli, M. & Maratea, A. A "Fuzzy Extension of Some Classical Concordance Measures and an Efficient Algorithm
    for Their Computation" Knowledge-Based Intelligent Information and Engineering Systems,
    Springer Berlin Heidelberg, 2008, 5179, 755-763

    :param labels_true:
    :param labels_pred:
    :return:
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    contingency = contingency_matrix(labels_true, labels_pred)

    cc = np.sum(contingency * contingency)
    N11 = (cc - n_samples)
    c1 = contingency.sum(axis=1)
    N01 = np.sum(c1 * c1) - cc
    c2 = contingency.sum(axis=0)
    N10 = np.sum(c2 * c2) - cc

    return (N11*1.0)/(N11+N01+N10)


def folkes_mallow_score(labels_true, labels_pred):
    """
    Folkes&Mallow score  computed according to:

    Ceccarelli, M. & Maratea, A. A "Fuzzy Extension of Some Classical Concordance Measures and an Efficient Algorithm
    for Their Computation" Knowledge-Based Intelligent Information and Engineering Systems,
    Springer Berlin Heidelberg, 2008, 5179, 755-763

    :param labels_true:
    :param labels_pred:
    :return:
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    contingency = contingency_matrix(labels_true, labels_pred)

    cc = np.sum(contingency * contingency)
    N11 = (cc - n_samples)
    c1 = contingency.sum(axis=1)
    N01 = np.sum(c1 * c1) - cc
    c2 = contingency.sum(axis=0)
    N10 = np.sum(c2 * c2) - cc

    return (N11*1.0)/np.sqrt((N11+N01)*(N11+N10))

def bhargavi_gowda_score(X, labels):
    """
    Score from:

    Bhargavi, M. & Gowda, S. D. "A novel validity index with dynamic cut-off for determining true clusters"
    Pattern Recognition , 2015, 48, 3673 - 3687

    :param X:
    :param labels:
    :return:
    """
    llabels = np.unique(labels)
    poslabels = maplabels(llabels)

    nclust = len(llabels)
    nex = len(labels)

    # Centroid of the data

    centroid = np.zeros((1, X.shape[1]))
    centroid += np.sum(X, axis=0)
    centroid /= X.shape[0]

    # Compute SSB and intracluster distance
    ccentroid = np.zeros((nclust, X.shape[1]))
    dist = 0.0
    for idx in llabels:
        center = np.zeros((1, X.shape[1]))
        center_mask = labels == idx
        center += np.sum(X[center_mask], axis=0)
        center /= center_mask.sum()
        ccentroid[poslabels[idx]] = center
        dvector = euclidean_distances(centroid.reshape(1, -1), ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
        dist += dvector.sum() * center_mask.sum()

    SSB = dist / len(labels)

    # Compute SSW
    dist = 0.0
    Intra = 0.0
    for idx in llabels:
        center_mask = labels == idx
        dvector = euclidean_distances(X[center_mask], ccentroid[poslabels[idx]].reshape(1, -1), squared=True)
        dist += dvector.sum()
        sdvector = euclidean_distances(X[center_mask], ccentroid[poslabels[idx]].reshape(1, -1), squared=False)
        Intra += sdvector.sum()

    SSW = dist / len(labels)

    SST = SSB + SSW


    # Centroids distance matrix
    cdistances = euclidean_distances(ccentroid, squared=False)

    Inter = np.sum(cdistances)/(nclust**2)


    return(np.abs((SSW/SSB)*SST) - (Intra/Inter) - (nex - nclust))