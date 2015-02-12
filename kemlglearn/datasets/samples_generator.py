"""
.. module:: samples_generator

samples_generator
*************

:Description: samples_generator

    

:Authors: bejar
    

:Version: 

:Created on: 21/01/2015 9:02 

"""

__author__ = 'bejar'

from rpy2.robjects.packages import importr
import numpy as np

def cluster_generator(n_clusters=3, sepval=0.5, numNonNoisy=5, numNoisy=0, numOutlier=0,
                      clustszind=2, clustSizeEq=100, rangeN=[100, 150], rotateind=True):
    """
    Generates clusters using the R package CusterGeneration
    See the documentation of that package for the meaning of the parameters

    You must have an R installation with the clusterGeneration package

    :param n_clusters:
    :param sepval:
    :return:
    """
    clusterG = importr('clusterGeneration')

    params = {'numClust': n_clusters,
          'sepVal': sepval,
          'numNonNoisy': numNonNoisy,
          'numNoisy': numNoisy,
          'numOutlier': numOutlier,
          'numReplicate': 1,
          'clustszind': clustszind,
          'clustSizeEq': clustSizeEq,
          'rangeN': rangeN,
          'rotateind': rotateind,
          'outputDatFlag': False,
          'outputLogFlag': False,
          'outputEmpirical': False,
          'outputInfo': False
         }

    x= clusterG.genRandomClust(**params)
    # nm = np.array(x[2][0].colnames)
    # nm = np.concatenate((nm, ['class']))
    m = np.matrix(x[2][0])
    v = np.array(x[3][0])
    v.resize((len(x[3][0])))
    #m = np.concatenate((m, v), axis=1)
    return  m, v
