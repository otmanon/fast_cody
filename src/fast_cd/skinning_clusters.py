import numpy as np
from sklearn.cluster import KMeans

from .average_onto_simplex import average_onto_simplex

'''
Clusters skinning weights


Inputs:
W - n x b skinning weights
D - b x 1 eigenvalue with each skinning weight
T - f x 3(4) tet geometry
k - number of clusters

Optional
l - power to raise D to
num_clustering_features - number of features to use for clustering (default 10)

:returns
l - f x 1 cluster label for each tet
'''

def skinning_clusters(W, D, T, k, l=2, num_clustering_features=10,
                      return_centroids=False, return_simplex_features=False):

    num_clustering_features = min(num_clustering_features, W.shape[1])
    # need to average the skinning weights over each tet
    assert(T.shape[1] == 4, "only tets implemented so far for clustering")

    Wt = average_onto_simplex(W, T)
    # Wt2 = Wt / np.power(D, 2)
    Wt = Wt / np.power(D, l)
    Wt = Wt[:, 0:num_clustering_features]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Wt)
    l = kmeans.labels_

    # igl.connected_components()

    if return_simplex_features:
        return l, Wt
    if return_centroids == True:
        return l, kmeans.cluster_centers_
    else:
        return l