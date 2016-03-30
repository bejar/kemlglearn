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
from kemlglearn.metrics import within_scatter_matrix_score, between_scatter_matrix_score, calinski_harabasz_score,\
    zhao_chu_franti_score, scatter_matrices_scores, davies_bouldin_score, variation_of_information_score, \
    jaccard_score, JaccardArandFolkes, folkes_mallow_score
from kemlglearn.cluster.consensus import SimpleConsensusClustering
from sklearn.metrics.cluster import normalized_mutual_info_score

#X, y_data = make_blobs(n_samples=100, n_features=10, centers=3, random_state=2)
#X = load_iris()['data']
X, y_data = make_circles(n_samples=1000, noise=0.5, random_state=4, factor=0.5)


# nc = 12
# _, X = cluster_generator(n_clusters=nc, sepval=0.01, numNonNoisy=15, numNoisy=3, rangeN=[50, 100])
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

# gkm = GlobalKMeans(n_clusters=nc)
# gkm.fit(X)
# print DaviesBouldin(X, gkm.labels_)
# print scatter_matrices_scores(X, gkm.labels_, ['Inertia'])

#
# print gkm.cluster_centers_.shape[0], gkm.inertia_
# print within_scatter_matrix_score(X, gkm.labels_)
# print between_scatter_matrix_score(X, gkm.labels_)


# fig = plt.figure()
#
# # ax = fig.gca(projection='3d')
# # pl.scatter(X[:, 1], X[:, 2], zs=X[:, 0], c=gkm.labels_, s=25)
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:, 0], X[:, 1], c=gkm.labels_)
#
# plt.show()
# for nc in range(2, 16):
#     km = KMeans(n_clusters=nc)
#     km.fit(X)
#     #print km.cluster_centers_.shape[0], km.inertia_
#     print nc, scatter_matrices_scores(X, km.labels_, ['CH', 'ZCF', 'Hartigan', 'Xu'])
#     print DaviesBouldin(X, km.labels_)

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=km.labels_)
#
# plt.show()

km = KMeans(n_clusters=3)
km.fit(X)

# print variation_of_information_score(km.labels_, y_data)
# print normalized_mutual_info_score(km.labels_, y_data)
print(jaccard_score(km.labels_, y_data))
print(folkes_mallow_score(km.labels_, y_data))
print(JaccardArandFolkes(km.labels_, y_data))

# print DaviesBouldin(X, km.labels_)
# print scatter_matrices_scores(X, km.labels_, ['Inertia'])

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=km.labels_)
#
# plt.show()


# simple = SimpleConsensusClustering(n_clusters=nc, n_components=40)
# simple.fit(X)
#
# print DaviesBouldin(X, simple.labels_)
# print scatter_matrices_scores(X, simple.labels_, ['Inertia'])

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:,0],X[:,1],c=simple.labels_)
#
# plt.show()


# simple = SimpleConsensusClustering(n_clusters=nc, n_components=40, consensus2='spectral')
# simple.fit(X)
#
# print DaviesBouldin(X, simple.labels_)
# print scatter_matrices_scores(X, simple.labels_, ['Inertia'])
#

# fig = plt.figure()
#
# ax = fig.add_subplot(111)
# plt.scatter(X[:, 0], X[:, 1], c=simple.labels_)
#
# plt.show()
