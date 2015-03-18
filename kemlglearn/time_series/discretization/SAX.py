"""
.. module:: SAX

SAX
*************

:Description: SAX

    

:Authors: bejar
    

:Version: 

:Created on: 18/03/2015 13:49 

"""

__author__ = 'bejar'


import numpy as np
from scipy.stats import norm

class SAX():
    """
    Sax representation of the time series
    """
    winlen = None
    step = None
    wrdlen = None
    voc = None
    intervals = None

    def __init__(self, window_length=100, step=1, word_length=10, voc_size=3):
        self.winlen = window_length
        self.step = step
        self.wrdlen = word_length
        self.voc = voc_size
        self.intervals = norm.ppf(np.arange(self.voc)/float(self.voc))


    # Todo: Check that this works
    def transform(self, X):
        """
        Computes the SAX representation for a vector of data
        The data is normalized before transformation

        Beware: If length is not a divisor of the vector size, some data
        points at the end will be ignored

        The intervals for the discretization are computed on every call
        :param length: Length of the wore
        :param voc: Length of the vocabulary
        :param intervals: list with the breakpoints of the discretization
                          if the parameter does not exist the intervals are
                          computed, but this is inefficient
        :return: a vector with length values in the range [-voc//2,voc//2]
        """

        res = np.zeros((X.shape[0]//self.step-self.winlen, self.wrdlen))

        for w in range(X.shape[0]//self.winlen):
            chunk = X[w*self.step, (w*self.step) + self.winlen]
            res[w] = self._SAX_function(chunk, self.wrdlen, self.voc, self.intervals)
        return res

    @staticmethod
    def _SAX_function(self, data, length, voc, intervals):
        """
        Computes the SAX representation for a vector of data
        The data is normalized before transformation

        Beware: If length is not a divisor of the vector size, some data
        points at the end will be ignored

        The intervals for the discretization are computed on every call
        :param length: Length of the wore
        :param voc: Length of the vocabulary
        :param intervals: list with the breakpoints of the discretization
                          if the parameter does not exist the intervals are
                          computed, but this is inefficient
        :return: a vector with length values in the range [-voc//2,voc//2]
        """
        index = np.zeros(length)
        data -= data.mean(0)
        data = np.nan_to_num(data / data.std(0))
        step = data.shape[0] / length
        for i in range(length):
            mr = np.sum(data[i*step:(i*step)+step])/length
            j = voc - 1
            while mr < intervals[j]:
                j -= 1
            index[i] = j - (voc / 2)
        return index
