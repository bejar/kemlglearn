"""
.. module:: Discretizer

Discretizer
*************

:Description: Discretizer

    

:Authors: bejar
    

:Version: 

:Created on: 13/03/2015 16:15 

"""

__author__ = 'bejar'

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


class Discretizer(BaseEstimator, TransformerMixin):
    """
    Discretization of the attributes of a dataset (unsupervised)

    method = 'equal' equal sized bins
             'frequency' bins with the same number of examples
    bins = number of bins
    """
    intervals = None

    def __init__(self, method='equal', bins=2):
        self.method = method
        self.bins = bins


    def fit(self,X):
        """
        Computes the discretization intervals

        :param X:
        :return:
        """
        if self.method == 'equal':
            self._fit_equal(X)
        elif self.method == 'frequency':
            self._fit_frequency(X)


    def _fit_equal(self, X):
        """
        Computes the discretization intervals for equal sized discretization

        :param X:
        :return:
        """
        self.intervals = np.zeros((self.bins, X.shape[1]))

        for i in range(X.shape[1]):
            vmin = np.min(X[:, i])
            vmax = np.max(X[:, i])

            step = np.abs(vmax - vmin) / float(self.bins)
            for j in range(self.bins):
                vmin += step
                self.intervals[j, i] = vmin

            self.intervals[self.bins-1, i] += 0.00000000001

    def _fit_frequency(self, X):
        """
        Computes the discretization intervals for equal frequency

        :param X:
        :return:
        """

        self.intervals = np.zeros((self.bins, X.shape[1]))

        quant = X.shape[0] / float(self.bins)
        for i in range(X.shape[1]):
            lvals = sorted(X[:, i])
            nb = 0
            while nb < self.bins:
                self.intervals[nb, i] = lvals[int((quant*nb) + quant)-1]
                nb += 1
            self.intervals[self.bins-1, i] += 0.00000000001

    def transform(self, X, copy=False):
        """
        Discretizes the attributes of a dataset
        :param X:
        :return:
        """
        if self.intervals is None:
            raise Exception('Discretizer: Not fitted')
        if copy:
            Y = X.copy()
        else:
            Y = X

        self._transform(Y)

        return Y

    def __discretizer(self, v, at):
        """

        :param v:
        :return:
        """
        i=0
        while i< self.intervals.shape[0] and v > self.intervals[i, at]:
            i += 1
        return i

    def _transform(self, X,):
        """

        :param X:
        :return:
        """
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                X[j, i] = self.__discretizer(X[j, i], i)

    def fit_transform(self, X, copy=False):
        """
        Fits and transforms the data

        :param X:
        :param copy:
        :return:
        """
        self.fit(X)
        return self.transform(X, copy)

