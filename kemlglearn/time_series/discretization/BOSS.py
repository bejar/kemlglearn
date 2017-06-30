"""
.. module:: BOSS

BOSS
*************

:Description: BOSS

    

:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 13:48 

"""


import numpy as np
from kemlglearn.preprocessing import Discretizer
import seaborn as sn
from collections import Counter
from kemlglearn.time_series.decomposition.MFT import mft
__author__ = 'bejar'


def boss_distance(histo1, histo2):
    """
    BOSS distance between two histograms

    Not really a distance because it is not symmetric

    :param histo1:
    :param histo2:
    :return:
    """
    val = 0
    for w in histo1:
        if w in histo2:
            val += (histo1[w] - histo2[w]) ** 2
        else:
            val += histo1[w] * histo1[w]

    return val


def euclidean_distance(histo1, histo2):
    """
    Euclidean distance between two histograms

    :param histo1:
    :param histo2:
    :return:
    """
    val = 0
    lnot = []
    for w in histo1:
        if w in histo2:
            val += (histo1[w] - histo2[w]) ** 2
            lnot.append(w)
        else:
            val += histo1[w] * histo1[w]

    for w in histo2:
        if w not in lnot:
            val += histo2[w] * histo2[w]

    return val


def cosine_similarity(histo1, histo2):
    """
    Euclidean distance between two histograms

    :param histo1:
    :param histo2:
    :return:
    """
    val = 0.0
    norm1 = 0.0
    for w in histo1:
        if w in histo2:
            val += (histo1[w] * histo2[w])
        norm1 += histo1[w] ** 2
    norm2 = 0.0
    for w in histo2:
        norm2 += histo2[w] ** 2

    return val / (np.sqrt(norm1) * np.sqrt(norm2))


def hamming_distance(histo1, histo2):
    """
    Euclidean distance between two histograms

    :param histo1:
    :param histo2:
    :return:
    """
    val = 0
    lnot = []
    for w in histo1:
        if w in histo2:
            val += np.abs((histo1[w] - histo2[w]))
            lnot.append(w)
        else:
            val += histo1[w]

    for w in histo2:
        if w not in lnot:
            val += histo2[w]

    return val


def bin_hamming_distance(histo1, histo2):
    """
    jaccard distance between two histograms

    :param histo1:
    :param histo2:
    :return:
    """
    s1 = set(histo1.keys())
    s2 = set(histo2.keys())
    return len(s1) + len(s2) - len(s1.intersection(s2))


class Boss():
    """
    Computes the BOSS words for a dictionary of series
    """

    def __init__(self, dseries, sampling, butfirst=False):
        """

        :param dseries:
        :param sampling:
        """
        self.series = dseries
        self.sampling = sampling
        self.coefs = {}
        self.codes = {}
        self.butfirst = butfirst

    def discretization_intervals(self, ncoef, wsize, vsize):
        """
        Computes the BOSS discretization for the signals, the word length is 2*ncoefs (real and imaginary part) except
        is there are coefficients that are zero

        :param ncoef:
        :param wsize:
        :return:
        """

        all_coefs = []
        for s in self.series:
            coefs = mft(self.series[s], self.sampling, ncoef, wsize, butfirst=self.butfirst)
            lcoefs = []
            for i in range(coefs.shape[1]):
                lcoefs.append(coefs[:, i].real)
                lcoefs.append(coefs[:, i].imag)

            all_coefs.append(np.stack(lcoefs, axis=-1))
            self.coefs[s] = all_coefs[-1]

        X = np.concatenate(all_coefs)

        self.disc = Discretizer(method='frequency', bins=vsize)
        self.disc.fit(X)

    def discretize(self):
        """
        Computes the words for each time series
        :param series:
        :return:
        """
        vocabulary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        def word(vec):
            """

            :param v:
            :return:
            """
            w = ''
            for v in vec:
                w += vocabulary[int(v)]
            return w

        for c in self.coefs:
            sdisc = self.disc.transform(self.coefs[c], copy=True).real
            prevw = word(sdisc[0])
            lvoc = [prevw]
            for i in range(1, sdisc.shape[0]):
                nword = word(sdisc[i])
                if nword != prevw:
                    lvoc.append(nword)
            self.codes[c] = Counter(lvoc)

            # print(c, self.codes[c])


if __name__ == '__main__':
    pass
    # wlen = 64
    # voclen = 3
    # ncoefs = 3
    #

    #
    # boss = Boss(dseries, 10)
    # boss.discretization_intervals(ncoefs, wlen, voclen)
    # boss.discretize()
