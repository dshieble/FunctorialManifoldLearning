import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from collections import defaultdict
from sklearn.manifold import spectral_embedding, MDS
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.decomposition import PCA
from numba import njit, jit
from numba import types
from numba.typed import Dict
from scipy.cluster.hierarchy import linkage
from helpers import mds, write_embedding_to_text_file, write_embedding_to_two_text_files, is_numeric
from sklearn.neighbors import NearestNeighbors
from sequence import mutate_sequence, generate_mutation_chain, sequence_distance, evaluate_embeddings


@jit(nopython=True)
def _build_distance_matrix_from_linked_helper(distance_matrix, cluster_tree, int_array=types.int64[:]):
    num_points = len(cluster_tree) + 1
    cluster_indices_to_points = Dict.empty(
        key_type=types.int64,
        value_type=int_array
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
    distance_matrix = np.zeros((len(cluster_tree) + 1, len(cluster_tree) + 1))
    return _build_distance_matrix_from_linked_helper(distance_matrix=distance_matrix, cluster_tree=cluster_tree)


def build_single_linkage_simplex_strength_matrix(condensed_raw_distance_matrix):
    cluster_tree = linkage(condensed_raw_distance_matrix)
    single_linkage_distance_matrix, _ = build_distance_matrix_from_linked(cluster_tree=cluster_tree)
    single_linkage_simplex_strength_matrix = np.exp(-single_linkage_distance_matrix)
    return single_linkage_simplex_strength_matrix


def single_linkage_mds_from_condensed(condensed_raw_distance_matrix, n_components):
    single_linkage_simplex_strength_matrix =  build_single_linkage_simplex_strength_matrix(condensed_raw_distance_matrix)
    return mds(single_linkage_simplex_strength_matrix, n_components=n_components)


def single_linkage_spectral_from_condensed(condensed_raw_distance_matrix, n_components, drop_first=False):
    single_linkage_simplex_strength_matrix =  build_single_linkage_simplex_strength_matrix(condensed_raw_distance_matrix)
    return spectral_embedding(single_linkage_simplex_strength_matrix, n_components=n_components, drop_first=drop_first)


def maximal_linkage_mds_from_condensed(condensed_distance_matrix, n_components):
    distance_matrix = squareform(condensed_distance_matrix)
    simplex_strength_matrix = np.exp(-distance_matrix)
    return mds(simplex_strength_matrix, n_components=n_components)


def maximal_linkage_spectral_from_condensed(condensed_distance_matrix, n_components, drop_first=False):
    distance_matrix = squareform(condensed_distance_matrix)
    simplex_strength_matrix = np.exp(-distance_matrix)
    return spectral_embedding(simplex_strength_matrix, n_components=n_components, drop_first=drop_first)




