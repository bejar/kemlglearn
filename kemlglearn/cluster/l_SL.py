from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from Leader import Leader

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
import numpy as np

import seaborn as sns
sns.set()


class LeadersSingleLink(object):
    """
    Class for Computing the leaders-Single-Link Algorithm (l-SL)

    Distance based clustering method to find arbitrary shaped clusters in a
    large dataset. First leaders clustering method is applied to a dataset to
    derive a set of leaders subsequently single-link method (with distance
    criteria) is applied to the leaders set to obtain final clustering.

    Reference:
    Bidyut Kr. Patra, Sukumar Nandi, and P. Viswanath. 2011.
    A distance based clustering method for arbitrary shaped clusters in large
    datasets. Pattern Recogn. 44, 12 (December 2011), 2862-2870.
    https://pdfs.semanticscholar.org/f46d/f301f493055ca374d81a9639690ad4bce50e.pdf
    """

    def __init__(self, h=None):
        """
        Instantiate l-SL with the minimum intra-cluster distance parameter (h)
        When set to none, it will be estimated when the dataset is fit
        :param h: float, distance threshold
        """
        self.h = h

    def fit(self, X):
        """
        Clusters the dataset X following Algorithm 2 (see reference)

        :param X: numeric matrix, (M_samples X N_features)
        """

        if not self.h:
            self.h = self._estimate_h(X)

        self.ld = Leader(radius=self.h / 2)

        followers = self.ld.fit_predict(X).astype('int')
        leaders = self.ld.cluster_centers_

        self.leaders_sl = self._single_linkage(leaders, self.h)

        self.labels = self.leaders_sl[followers]

    def fit_predict(self, X):
        """
        Runs fit methods and return predicted clusters

        :param X: numeric matrix, (M_samples X N_features)
        :return: numeric vector of cluster index (M_samples X 1)
        """
        self.fit(X)
        return self.labels

    def predict(self, X):
        """
        Given a new set of values, runs a prediction without refitting the
        model. Note that if the sample forms a new cluster, the numeric label
        '-1' is returned.
        :param X: numeric matrix, (M_samples X N_features)
        :return: numeric vector of cluster index based on existing clusters
          (M_samples X 1)
        """
        followers = np.array(self.ld.predict(X), dtype=int)
        unknown_clus = followers < 0
        labels = self.leaders_sl[followers]
        labels[unknown_clus] = -1
        return labels

    def _single_linkage(self, D, h):
        """
        Perform hierarchical/agglomerative clustering based on Single Linkage
         and euclidean distance metric.
        :param D: numeric matrix to be clustered, (M_samples X N_features)
        :param h: distance used to cut the tree
        :return: numeric vector of cluster index (M_samples X 1)
        """
        sl = linkage(y=D, method="single", metric="euclidean")
        cluster = cut_tree(sl, height=h)
        return cluster[:, 0]

    def _estimate_h(self, X):
        """
        Estimate the minimum intra-cluster distance parameter (h). Method based
        on the second approach mentioned in the reference which is the
        maximum life time clustering

        :param X: numeric matrix, (M_samples X N_features)
        :return: numeric value, estimation of h (maximum life time)
        """
        sample_idx = np.random.randint(X.shape[0],
                                       size=np.int(np.sqrt(X.shape[0])))
        sample_X = X[sample_idx, :]
        sl = linkage(y=sample_X, method="single", metric="euclidean")
        life_time = np.diff(sl[:, 2])
        return np.max(life_time)


# l-SL Synthetic Data (1M)
import time
tic = time.time()

moons_data, moons_labels = make_moons(n_samples=1000000, noise=.05)
lSL = LeadersSingleLink()
ypred = lSL.fit_predict(moons_data)

toc = time.time()
print toc-tic
# 102.523000002

cmap = sns.color_palette(n_colors=np.unique(ypred).size)
colors = np.array(cmap)[ypred]
plt.scatter(moons_data[:, 0], moons_data[:, 1], c=colors, alpha=0.2)
plt.title("leader-Single-Link Clustering: Synthetic Data (1M)")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("lSL_synthetic.png")


# l-SL NCI60 Data (1M)
from sklearn.metrics import adjusted_rand_score

nci_data = pd.read_csv("https://raw.githubusercontent.com/hyunblee/"
                       "ISLR-with-Python/master/Data/NCI60_data.csv",
                       header=0, index_col=0)
nci_labs = pd.read_csv("https://raw.githubusercontent.com/hyunblee/"
                       "ISLR-with-Python/master/Data/NCI60_labs.csv",
                       header=0, index_col=0)

# # Labels to Numeric Vector of clusters
# ylabs = nci_labs.values[:, 0]
# labs_map = dict(zip(np.unique(ylabs), np.arange(np.unique(ylabs).size)))
# y = np.array([labs_map[lab] for lab in ylabs])

# Agglomerative Clustering as ground truth
plt.figure(figsize=(16,8))
hc = linkage(y=nci_data.values, method="complete", metric='euclidean')
dendrogram(hc, labels=nci_labs.x.values, leaf_rotation=0, leaf_font_size=8,
           orientation='left')
plt.savefig("dendrogram.png")

y = cut_tree(hc, n_clusters=4)[:, 0]


# Look for h parameter
lSL = LeadersSingleLink()
h_range = np.arange(15, 105, 5)
num_clus = []
ar_scores = []
for hh in h_range:
    lSL.h = hh
    ypred = lSL.fit_predict(nci_data.values)

    num_clus.append(np.unique(ypred).size)
    ar_scores.append(adjusted_rand_score(y, ypred))


# Plot
sns.set_style('ticks')
cmap = sns.color_palette(n_colors=3)

fig, ax1 = plt.subplots()
ax1.plot(h_range, num_clus, marker="s", color=cmap[0])
ax1.set_xlabel("parameter $h$ (min inter-cluster dist)")
ax1.set_ylabel('Num. of clusters')
ax1.tick_params('y', colors=cmap[0])
ax1.set_xticks(h_range)
ax1.set_yticks(num_clus)

ax2 = ax1.twinx()
ax2.plot(h_range, ar_scores, linestyle=":", color=cmap[2])
ax2.set_ylabel('Adjusted Rand Score')
ax2.tick_params('y', colors=cmap[2])
# ax2.set_yticks(ar_scores)

plt.title("leader-Single-Link parameter search: NCI60 dataset")
# plt.show()
plt.savefig("lSL_NCI60.png")
