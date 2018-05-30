import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


def clus_gap(X, k_max, B=100, space_H0="scaledPCA"):
    """
    Gap Statistic for Estimating the Number of Clusters
    Uses k-means as clustering method and calculates distances using squared
    euclidean distances.

    Example of usage:
        from sklearn import datasets

        iris = datasets.load_iris()
        cg = clus_gap(iris.data, 10)
        print cg
            [[5.83015724 5.9186501  0.08849287 0.06755358 0.        ]
             [4.3331561  4.93657725 0.60342114 0.04290971 0.        ]
             [3.67555155 4.54973961 0.87418806 0.04030847 0.        ]
             [3.35546532 4.35678088 1.00131556 0.04102972 0.        ]
             [3.14707004 4.19149574 1.0444257  0.04051674 1.        ]
             [2.97495494 4.04003479 1.06507986 0.04210669 0.        ]
             [2.84222445 3.90765753 1.06543308 0.04127175 0.        ]
             [2.70408911 3.79704167 1.09295257 0.04157361 0.        ]
             [2.64400663 3.70841689 1.06441026 0.04013879 0.        ]
             [2.58567649 3.6346142  1.04893771 0.03954204 0.        ]]

        clus_gap_plot(cg)

    Tested with R results:
        R> library(cluster)
        R> clusGap(iris[,1:4], FUN=kmeans, K.max=10, d.power=2)

    Reference:
        Tibshirani, R., Walther, G. and Hastie, T. (2001)
        Estimating the number of data clusters via the Gap statistic.
        Journal of the Royal Statistical Society B, 63, 411-423
        (https://web.stanford.edu/~hastie/Papers/gap.pdf)

    :param X: numeric matrix (N_samples X M_features)
    :param k_max: integer, maximum number of cluster partitions to consider
    :param B: integer, number of bootstrap samples. Default B = 100
    :param space_H0: string specifying the space of the null hypothesis of no
        clusters. One of "scaledPCA" or "original". Both use an uniform
        distribution. Default: "scaledPCA"

    :return: numeric maxtrix with k_max rows and 5 columns with the following
        variables: logW, ElogW, gap, SEsim, cut. SEsim is the standard error of
        gap (s_k) computed from std. dev. of the bootstrapped gap values.
        gap = ElogW - logW. Cut is a binary variable indicating the estimated
        number of clusters.

    """
    # Check input data
    assert X.ndim == 2, "'X' dimension must be 2"
    assert k_max >= 2, "'k_max' must be at least 2 or greater"
    assert X.shape[0] >= 1, "No samples detected"
    assert X.shape[1] >= 1, "No features detected"
    assert B > 0, "'B' has to be a positive integer"

    def Wk(X, kk):
        if kk > 1:
            model = KMeans(n_clusters=kk)
            clus = model.fit_predict(X)
        else:
            clus = np.zeros(X.shape[0], dtype=int)

        Dk = np.empty(kk)

        for k in np.arange(kk):
            xs = X[clus == k, :]
            Dk[k] = np.sum(np.tril(euclidean_distances(xs, squared=True))
                           / xs.shape[0])

        return 0.5 * np.sum(Dk)

    logW = np.zeros(k_max)
    for k in np.arange(k_max):
        logW[k] = np.log(Wk(X, k + 1))

    xs = scale(X, with_mean=True, with_std=False)
    if space_H0 == "scaledPCA":
        _, _, vh = np.linalg.svd(xs, full_matrices=True)
        xs = np.dot(xs, np.transpose(vh))

    rng_x1 = np.array([np.min(xs, axis=0),
                       np.max(xs, axis=0)])

    logWks = np.zeros((B, k_max))
    for b in np.arange(B):
        z1 = np.zeros_like(X)
        for c in np.arange(X.shape[1]):
            z1[:, c] = np.random.uniform(low=rng_x1[0, c], high=rng_x1[1, c],
                                         size=X.shape[0])
        if space_H0 == "scaledPCA":
            z = np.dot(z1, vh)
        else:
            z = z1
        z = np.add(z, np.mean(X, axis=0))

        for k in np.arange(k_max):
            logWks[b, k] = np.log(Wk(z, k + 1))

    ElogW = np.mean(logWks, axis=0)
    SEsim = np.sqrt((1+1/np.float(B)) * np.var(logWks, axis=0))
    gap = ElogW - logW

    cut = np.zeros(k_max)
    cut_idx = np.where(gap[:-1] >= (gap[1:] - SEsim[1:]))[0]
    if cut_idx.size > 0:
        cut[cut_idx[0]] = 1

    return np.transpose(np.vstack([logW, ElogW, gap, SEsim, cut]))


def clus_gap_plot(cg_out):
    k = np.arange(cg_out.shape[0])+1
    gap = cg_out[:, 2]
    se = cg_out[:, 3]
    cut = np.where(cg_out[:, 4])[0] + 1

    plt.errorbar(k, gap, yerr=se, fmt='-o')
    plt.xlabel("Number of clusters $k$")
    plt.xticks(k)
    plt.ylabel("$Gap_k$")
    plt.title("Gap Statistic for Estimating the Number of Clusters")
    if cut.size > 0:
        plt.axvline(x=cut[0], linestyle='--', color='red')
        plt.legend(["Estimated num. of clusters", "Gap"])

    # plt.show()


# Test Implementation
from sklearn import datasets
iris = datasets.load_iris()
cg = clus_gap(iris.data, 10)
clus_gap_plot(cg)
plt.show()

# Report
from amltlearn.datasets import make_blobs
from amltlearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import silhouette_score
import seaborn as sns
sns.set()

# # Synthetic Data
# blobs, blabels = make_blobs(n_samples=200, n_features=3,
#                             centers=[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
#                             cluster_std=[0.2, 0.1, 0.3])
# data = blobs
# n = 5

# # Iris Dataset
# iris = datasets.load_iris()
# data = iris.data
# n = 10

# NCI60 Dataset
import pandas as pd
tmp = pd.read_csv("https://raw.githubusercontent.com/hyunblee/"
                  "ISLR-with-Python/master/Data/NCI60_data.csv")
data = tmp.iloc[:, 1:].values
n = 8

lscores = []
nclusters = range(2, n+1)
for nc in nclusters:
    km = KMeans(n_clusters=nc, n_init=10, random_state=0)
    labels = km.fit_predict(data)
    lscores.append((
        silhouette_score(data, labels),
        calinski_harabasz_score(data, labels),
        davies_bouldin_score(data, labels)))


fig = plt.figure(figsize=(14,11))

ax = fig.add_subplot(2,1,1)
cg = clus_gap(data, n)
clus_gap_plot(cg)

ax = fig.add_subplot(2,3,4)
score = [x for x,_,_ in lscores]
plt.plot(nclusters, score)
plt.xlabel("Number of clusters")
plt.xticks(nclusters)
plt.ylabel("Score")
plt.title("Silhouette")
plt.axvline(x=nclusters[np.argmax(score)], linestyle='--', color='red')
plt.legend(["Score", "$\hat{k}$"])

ax = fig.add_subplot(2,3,5)
score = [x for _, x,_ in lscores]
plt.plot(nclusters, score)
plt.xlabel("Number of clusters")
plt.xticks(nclusters)
plt.ylabel("Score")
plt.title("Calinski and Harabaz")
plt.axvline(x=nclusters[np.argmax(score)], linestyle='--', color='red')
plt.legend(["Score", "$\hat{k}$"])

ax = fig.add_subplot(2,3,6)
score = [x for _, _, x in lscores]
plt.plot(nclusters, score)
plt.xlabel("Number of clusters")
plt.xticks(nclusters)
plt.ylabel("Score")
plt.title("Davies-Bouldin")
plt.axvline(x=nclusters[np.argmin(score)], linestyle='--', color='red')
plt.legend(["Score", "$\hat{k}$"])

plt.show()
