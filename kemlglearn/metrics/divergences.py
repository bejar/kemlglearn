"""
.. module:: distances

distances
*************

:Description: distances

    Distances and divergence functions for probability matrices and vectors

:Authors: bejar
    

:Version: 

:Created on: 17/11/2014 13:01 

"""

__author__ = 'bejar'

import numpy as np


def simetrized_kullback_leibler_divergence(m1, m2):
    """
    Simetrized Kullback-Leibler divergence between two probability matrices/vectors
    :param m1:
    :param m2:
    :return:
    """
    lm1 = np.log(m1)
    lm2 = np.log(m2)
    lquot12 = np.log(m1 / m2)
    lquot21 = np.log(m2 / m1)
    dkl12 = lm1 * lquot12
    dkl21 = lm2 * lquot21
    return dkl12.sum() + dkl21.sum()

def kullback_leibler_divergence(m1, m2):
    """
    Kullback-Leibler divergence between two probability matrices/vectors
    :param m1:
    :param m2:
    :return:
    """
    lm1 = np.log(m1)
    lm2 = np.log(m2)
    lquot12 = np.log(m1 / m2)
    lquot21 = np.log(m2 / m1)
    dkl12 = lm1 * lquot12

    return dkl12.sum()

def jensen_shannon_divergence(m1, m2):
    """
    Jensen Shannon Divergence between two probability matrices/vectors

    :param m1:
    :param m2:
    :return:
    """
    m = 0.5*(m1+m2)

    return (0.5 * kullback_leibler_divergence(m1, m)) + (0.5 * kullback_leibler_divergence(m2, m))


def renyi_half_divergence(m1, m2):
    """
    Renyi divergence for parameter 1/2 between two probability matrices/vectors
    :param m1:
    :param m2:
    :return:
    """

    pm = m1 * m2
    spm = np.sqrt(pm)

    return -2 * np.log(spm.sum())


def square_frobenius_distance(m1, m2):
    """
    Square frobenius distance between two probability matrices/vectors
    :param m1:
    :param m2:
    :return:
    """
    c = m1 - m2
    c = c * c
    return c.sum()


def bhattacharyya_distance(m1, m2):
    """
    Bhattacharyya distance between two probability matrices/vectors
    :param m1:
    :param m2:
    :return:
    """
    sum = 0.0
    for a, b in zip(m1, m2):
        sum += np.sum(np.sqrt(a*b))
    return - np.log(sum)

def hellinger_distance(m1, m2):
    """
    Bhattacharyya distance between two probability matrices/vectors
    :param m1:
    :param m2:
    :return:
    """
    sum = 0.0
    for a, b in zip(m1, m2):
        sum += np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)
    return (1/np.log(2)) * np.sqrt(sum)
