"""
.. module:: test

test
*************

:Description: test

    

:Authors: bejar
    

:Version: 

:Created on: 07/07/2014 11:12 

"""
__author__ = 'bejar'



from Leader import Leader
from GlobalKMeans import GlobalKMeans
from sklearn.datasets import make_blobs, load_iris, make_circles
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kemlglearn.datasets import cluster_generator
from kemlglearn.metrics import within_scatter_matrix_score, between_scatter_matrix_score, CalinskiHarabasz,\
    ZhaoChuFranti, scatter_matrices_scores, DaviesBouldin

#X, y_data = make_blobs(n_samples=1000, n_features=10, centers=20, random_state=2)
#X = load_iris()['data']
#X, y_data = make_circles(n_samples=1000, noise=0.5, random_state=4, factor=0.5)

_, X = cluster_generator(n_clusters=8, sepval=0.01, numNonNoisy=15, numNoisy=3, rangeN=[200, 250])
# ld = Leader(radius=25.0)
#
# #print timeit.timeit(stmt='ld.fit(X)',setup=setup,number=10)
# ld.fit(X)
#
# # print ld.cluster_centers_.shape[0]
#
# fig = plt.figure()
#
# # ax = fig.gca(projection='3d')
# # pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=ld.labels_, s=25)
# #
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=ld.labels_)
#
# plt.show()

# gkm = GlobalKMeans(n_clusters=5)
# gkm.fit(X)
#
# print gkm.cluster_centers_.shape[0], gkm.inertia_
# print within_scatter_matrix_score(X, gkm.labels_)
# print between_scatter_matrix_score(X, gkm.labels_)

#
# fig = plt.figure()
#
# # ax = fig.gca(projection='3d')
# # pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=gkm.labels_, s=25)
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:, 0], X[:, 1], c=gkm.labels_)
#
# plt.show()
for nc in range(2, 16):
    km = KMeans(n_clusters=nc)
    km.fit(X)
    #print km.cluster_centers_.shape[0], km.inertia_
    print nc, scatter_matrices_scores(X, km.labels_, ['CH', 'ZCF', 'Hartigan', 'Xu'])
    print DaviesBouldin(X, km.labels_)

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=km.labels_)
#
# plt.show()
