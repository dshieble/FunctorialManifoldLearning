import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import squareform, pdist, cdist
from numba import njit, jit
from numba import types
from numba.typed import Dict
from scipy.cluster.hierarchy import linkage
from sklearn.manifold import MDS

def mds(strength_matrix, n_components):
    """
    Args:
        strength_matrix: Matrix where i,j is the strength of the connection between row entity i and col entity j
        n_components: Number of dimensions
    Returns:
        Non-Metric Multidimensional Scaling embeddings
    """
    return MDS(n_components=n_components, dissimilarity="precomputed").fit_transform(-np.log(strength_matrix))


@jit(nopython=True)
def _build_distance_matrix_from_linked_helper(distance_matrix, cluster_tree, value_type=types.int64[:]):
    """
    Convert a distance matrix to a strength matrix describing the strength of the connections between points
        according to single linkage
    Args:
        distance_matrix: The output of scipy.spatial.distance.pdist
        cluster_tree: Output of scipy.cluster.hierarchy.linkage
    Returns:
        Embedding matrix
    """
    num_points = len(cluster_tree) + 1
    cluster_indices_to_points = Dict.empty(
        key_type=types.int64,
        value_type=value_type
    )
    for i in np.arange(num_points, dtype=np.int64):
        cluster_indices_to_points[i] = np.array([i], dtype=np.int64)
    for i in np.arange(num_points, num_points + len(cluster_tree), dtype=np.int64):
        cluster_indices_to_points[i] = np.array([np.int64(x) for x in range(0)])

    for i in range(len(cluster_tree)):
        l = cluster_tree[i]
        left_cluster = cluster_indices_to_points[int(l[0])]
        right_cluster = cluster_indices_to_points[int(l[1])]

        cluster_indices_to_points[num_points + i] = np.zeros(len(left_cluster) + len(right_cluster), dtype=np.int64)
        for il in range(len(left_cluster)):
            cluster_indices_to_points[num_points + i][il] = left_cluster[il]
        for ir in range(len(right_cluster)):
            cluster_indices_to_points[num_points + i][len(left_cluster) + ir] = right_cluster[ir]

        for left_point in left_cluster:
            for right_point in right_cluster:
                distance_matrix[left_point][right_point] = l[2]
                distance_matrix[right_point][left_point] = l[2]
    return distance_matrix, cluster_indices_to_points


def build_distance_matrix_from_linked(cluster_tree):
    """
    Given a clustering linkage tree, generate a distance matrix based on the tree depth
        at which points end up in the same cluster
    Args:
        cluster_tree: Output of scipy.cluster.hierarchy.linkage
    Returns:
        Distance matrix
    """
    distance_matrix = np.zeros((len(cluster_tree) + 1, len(cluster_tree) + 1))
    return _build_distance_matrix_from_linked_helper(distance_matrix=distance_matrix, cluster_tree=cluster_tree)


def build_single_linkage_simplex_strength_matrix(condensed_raw_distance_matrix):
    """
    Convert a distance matrix to a strength matrix describing the strength of the connections between points
        according to single linkage
    Args:
        condensed_raw_distance_matrix: The output of scipy.spatial.distance.pdist
    Returns:
        Embedding matrix
    """
    cluster_tree = linkage(condensed_raw_distance_matrix)
    single_linkage_distance_matrix, _ = build_distance_matrix_from_linked(cluster_tree=cluster_tree)
    single_linkage_simplex_strength_matrix = np.exp(-single_linkage_distance_matrix)
    return single_linkage_simplex_strength_matrix


def single_linkage_mds_from_condensed(condensed_raw_distance_matrix, n_components):
    """
    Generate single linkage scaling embeddings from a distance matrix
    Args:
        condensed_raw_distance_matrix: The output of scipy.spatial.distance.pdist
        n_components: The number dimensions
    Returns:
        Embedding matrix
    """
    single_linkage_simplex_strength_matrix =  build_single_linkage_simplex_strength_matrix(condensed_raw_distance_matrix)
    return mds(n_components=n_components, strength_matrix=single_linkage_simplex_strength_matrix)




def maximal_linkage_mds_from_condensed(condensed_raw_distance_matrix, n_components):
    """
    Generate MDS embeddings from a distance matrix
    Args:
        condensed_raw_distance_matrix: The output of scipy.spatial.distance.pdist
        n_components: The number dimensions
    Returns:
        Embedding matrix
    """
    distance_matrix = squareform(condensed_raw_distance_matrix)
    simplex_strength_matrix = np.exp(-distance_matrix)
    return mds(n_components=n_components, strength_matrix=simplex_strength_matrix)
