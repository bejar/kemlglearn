"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 22/01/2015 10:46 

"""

__author__ = 'bejar'

from .SimpleConsensusClustering import SimpleConsensusClustering
from .MeanPartition import MeanPartitionClustering

__all__ = ['SimpleConsensusClustering',
           'MeanPartitionClustering']
