import numpy as np


def cluster_centroids_spectral( B, cluster_indices):
    """Computes for clusters in spectral space.

    Parameters
    ----------
    B : (n, d) numpy float array
        d-dimensional spectral feature vectors.
    cluster_indices : (n, ) numpy int array
        Cluster indices.

    Returns
    -------
    centroids : (num_clusters, d) numpy float array
        Centroids in spectral space.
    """
    # Determine the number of clusters
    num_clusters = np.max(cluster_indices) + 1
    # Initialize an array to store the centroids
    centroids = np.zeros((num_clusters, B.shape[1]))
    # Calculate centroids for each cluster
    for cluster_idx in range(num_clusters):
        cluster_mask = (cluster_indices == cluster_idx)
        cluster_points = B[cluster_mask, :]
        centroid = np.mean(cluster_points, axis=0)
        centroids[cluster_idx] = centroid
    return centroids


def cluster_centroids_euclidean(positions, masses, cluster_indices):
    """ Computes euclidean centroids for a set of clusters.
    Parameters
    ----------
    positions : (n, d) numpy float array
        d-dimensional positions.
    masses : (n, ) numpy float array
        Masses/weights for each of the n-positions.
    cluster_indices : (n, ) numpy int array
        Cluster indices.

    Returns
    ------
    centroids : (num_clusters, d) numpy float array
        Centroids in euclidean space.
    """
    unique_clusters = np.unique(cluster_indices)
    centroids = np.zeros( (unique_clusters.shape[0] ,positions.shape[1] ))

    for cluster in unique_clusters:
        cluster_mask = (cluster_indices == cluster)
        cluster_positions = positions[cluster_mask]
        cluster_masses = masses[cluster_mask]

        weighted_sum = np.sum(cluster_positions * cluster_masses[:, np.newaxis], axis=0)
        total_mass = np.sum(cluster_masses)

        cluster_centroid = weighted_sum / total_mass
        centroids[cluster] =  cluster_centroid

    return centroids