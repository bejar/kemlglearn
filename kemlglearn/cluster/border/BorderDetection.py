"""
.. module:: BorderDetection

BorderDetection
*************

:Description: BorderDetection

    Algorithms for cluster border detection

:Authors: bejar
    

:Version: 

:Created on: 16/06/2016 9:25 

"""

__author__ = 'bejar'

from sklearn.neighbors import NearestNeighbors
import numpy as np

def QiuCaoBorder(X, n_neighbors, boundary_ratio, filter_ratio):
    """
    Compute the border points of the clusters from the dataset as described in:

    Qiu, B. & Cao, X. Clustering boundary detection for high dimensional space based on space
    inversion and Hopkins statistics Knowledge-Based Systems , 2016, 98, 216 - 225

    :return:
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn.fit(X)

    Xsym = np.zeros(X.shape[0])
    Filter = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        dnn, lnn = nn.kneighbors(X[i,:].reshape(1, -1))
        PLocations = np.zeros((n_neighbors, X.shape[1]))
        sdist = 0
        for j in range(n_neighbors):
            PLocations[j] = X[lnn[0][j+1]] - X[lnn[0][0]]
            sdist += np.exp(dnn[0][j+1]/n_neighbors)

        vsym = PLocations.sum(axis=1)
        Xsym[i] = np.abs(vsym).sum()
        Filter[i] = sdist

    eps1 = sorted(Xsym, reverse=True)[int(boundary_ratio*X.shape[0])]
    eps2 = sorted(Filter, reverse=True)[int(filter_ratio*X.shape[0])]

    border = []
    for i in range(X.shape[0]):
        if Xsym[i] > eps1 and Filter[i] < eps2:
            border.append(i)

    return border

if __name__ == '__main__':
    from sklearn.datasets import make_blobs, load_iris, make_circles
    import pylab as pl
    import matplotlib.pyplot as plt

    #X = load_iris()['data']

    X, y_data = make_blobs(n_samples=10000, n_features=5, centers=5, cluster_std=0.5, random_state=2)

    bd = QiuCaoBorder(X, 5, 0.2, 0.01)
    print(bd)
    labels = np.zeros(X.shape[0])+1
    for i in bd:
        labels[i] = 5
    fig = plt.figure()
    plt.scatter(X[:,0],X[:,1], s=labels*10)
    plt.show()









