"""
.. module:: Imputer

Imputer
*************

:Description: KnnImputer

    Class for the imputation of missing values using the k nearest neigbours


:Authors: bejar
    

:Version: 

:Created on: 01/09/2015 14:17 

"""

__author__ = 'bejar'


import numpy as np
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors

class KnnImputer(TransformerMixin):
    """
    Missing values imputation using the mean of the k-neighbors considering the
    dimensions that are not missing.

    It only uses the examples that do not have any missing value

    Parameters:

    missing_values: float or 'NaN'
     Value that indicates a missing value

    n_neighbors: int
       The number of neighbors to consider
    distance: str
       distance to use to compute the neighbors ('euclidean')
    """
    neigh = None
    miss_val = None
    dist = None
    miss_ind_ = None

    def __init__(self, missing_values='NaN', n_neighbors=1, distance='euclidean'):
        self.neigh = n_neighbors
        self.miss_val = float(missing_values)
        self.dist = distance

    def fit(self):
        """
        does nothing

        """

    def _transform(self, X):
        """
        Imputes the missings
        :param X:
        :return:
        """

        l_miss_ex = []
        l_no_miss_ex = []
        self.miss_ind_ = []
        for row in range(X.shape[0]):
            l_miss_att = []
            for column in range(X.shape[1]):
                if np.isnan(X[row, column]) or  X[row, column] ==self.miss_val:
                    l_miss_att.append(column)

            if l_miss_att:
                l_miss_ex.append((row, l_miss_att))
                self.miss_ind_.append(row)
            else:
                l_no_miss_ex.append(row)

        if not l_no_miss_ex:
            raise Exception('KnnImputer: All examples have missing values')
        else:
            nomiss = X[l_no_miss_ex]
            if nomiss.shape[0] < self.neigh:
                raise Exception('KnnImputer: Not enough examples without missings')
            for ex, att in l_miss_ex:
                l_sel = [s for s in range(X.shape[1]) if s not in att]
                knn = NearestNeighbors(n_neighbors=self.neigh, metric=self.dist)
                knn.fit(nomiss[:, l_sel])

                l_neigh = knn.kneighbors(X[ex][l_sel].reshape(1, -1), return_distance=False)[0]
                for a in att:
                    l_mean = nomiss[l_neigh, a]
                    X[ex][a] = np.mean(l_mean)
        return X

    def fit_transform(self, X, copy=True):
        """
        Looks for the examples with missing values and computes the new values

        :param matrix X: data matrix
        :param bool copy: If True returns a copy of the data
        :return:
        """
        if copy:
            y = X.copy()
        else:
            y = X

        self._transform(y)

        return y


if __name__ == '__main__':

    mean, cov = [0, 0, 0], [(1, .5, .5), (.5, 1, .5), (.5, .5, 1)]
    data = np.random.multivariate_normal(mean, cov, 200)
    vals = np.random.choice(200, size=20, replace=False)

    for v in vals[0:20]:
        data[v][0] = np.nan

    kimp = KnnImputer(n_neighbors=2)

    data2 = kimp.fit_transform(data)

    print (kimp.miss_ind_)
    for i in range(data.shape[0]):
        print (data[i], data2[i])