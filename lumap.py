from __future__ import print_function

from scipy.sparse import coo_matrix
from sklearn.manifold import spectral_embedding
from sklearn.utils import check_random_state
import numpy as np
import umap.distances as dist
from helpers import EUCLIDEAN, get_adjacency_matrix
from umap.umap_ import fuzzy_simplicial_set, simplicial_set_embedding, find_ab_params

SPECTRAL_INIT = "spectral"


def fit_umap(X, n_neighbors, metric, n_components=2):
    sparse_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        random_state=check_random_state(0),
        n_neighbors=n_neighbors,
        metric=metric)

    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return simplicial_set_embedding(
        data=X,
        graph=sparse_graph,
        n_components=n_components,
        initial_alpha=1.0,
        a=a,
        b=b,
        gamma=1.0,
        negative_sample_rate=5,
        n_epochs=0,
        init=SPECTRAL_INIT,
        random_state=check_random_state(0),
        metric=metric,
        metric_kwds={},
        output_metric=dist.named_distances_with_gradients[EUCLIDEAN],
        output_metric_kwds={},
        euclidean_output=(metric == EUCLIDEAN),
        parallel=False,
        verbose=False,
    )

def fit_lumap(X, n_neighbors, metric, n_components=2):
    """
    Build the fuzzy simplices UMAP-style (via fuzzy unions of local metric spaces) and then
        fit the matrix Laplacian Eigenmaps style (via graph laplacian)  
    """
    sparse_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        random_state=check_random_state(0),
        n_neighbors=n_neighbors,
        metric=metric)
    return spectral_embedding(sparse_graph, n_components=n_components)


def fit_mlce(X, n_neighbors, metric, n_components=2):
    """
    Maximal Linkage Cross Entropy

    Build the fuzzy simplices Laplacian Eigenmaps-style (via maximal linkage clustering and
        inverse -log fuzzy simplex weights) and then fit the matrix UMAP stype (via fuzzy cross entropy)
    """
    dense_graph = get_adjacency_matrix(X=X, n_neighbors=n_neighbors, metric=metric)
    graph = coo_matrix(dense_graph)

    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return simplicial_set_embedding(
        data=X,
        graph=graph,
        n_components=n_components,
        initial_alpha=1.0,
        a=a,
        b=b,
        gamma=1.0,
        negative_sample_rate=5,
        n_epochs=0,
        init=SPECTRAL_INIT,
        random_state=check_random_state(0),
        metric=metric,
        metric_kwds={},
        output_metric=dist.named_distances_with_gradients[EUCLIDEAN],
        output_metric_kwds={},
        euclidean_output=(metric == EUCLIDEAN),
        parallel=False,
        verbose=False,
    )
