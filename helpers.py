from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from tqdm import tqdm
import annoy
from annoy import AnnoyIndex
import random
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.manifold import spectral_embedding, MDS
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.decomposition import PCA

import numpy as np
from numba import njit, jit
from numba import types
from numba.typed import Dict
from scipy.cluster.hierarchy import linkage
from matplotlib.colors import BASE_COLORS


EUCLIDEAN = "euclidean"


def mds(A, n_components):
    return MDS(n_components=n_components).fit_transform(A)

def write_embedding_to_text_file(entity_ids, embeddings, fname):
    assert len(entity_ids) == len(embeddings)
    with open(fname, "w", encoding='utf-8') as f:
        for entity_id, embd in zip(entity_ids, embeddings):
            entries = "\t".join([entity_id] + [str(e) for e in embd])
            f.write(u"{}\n".format(entries))

def write_embedding_to_two_text_files(entity_ids, embeddings, entity_fname, embedding_fname):
    assert len(entity_ids) == len(embeddings)
    with open(entity_fname, "w", encoding='utf-8') as f:
        for entity_id in entity_ids:
            f.write(u"{}\n".format(entity_id))
    with open(embedding_fname, "w", encoding='utf-8') as f:
        for embd in embeddings:
            f.write(u"{}\n".format("\t".join([str(e) for e in embd])))

def is_numeric(n):
    out = True
    try:
        int(n)
    except Exception:
        out = False
    return out

def get_adjacency_matrix(X, n_neighbors, metric):
    """
    Compute the adjacency matrix of the graph that has an edge with weight e^{-d(xi,xj)} between vertices
        xi and xj. If n_neighbors is not None, then limit to the top N incoming and outgoing edges of each vertex
    """
    if n_neighbors is not None:
        dists = np.exp(-cdist(X, X, metric=metric))
        top_n_indices = np.transpose(np.argpartition(dists, -n_neighbors, axis=1)[-n_neighbors:])
        top_n_values = np.take_along_axis(dists, top_n_indices, axis=1)
        out = np.zeros(dists.shape)
        np.put_along_axis(
            arr=out,
            indices=top_n_indices,
            values=top_n_values,
            axis=1)
    else:
        out = np.exp(-cdist(X, X, metric=metric))
    return out

def fit_laplacian_eigenmaps(X, n_neighbors=20, metric=EUCLIDEAN, n_components=2):
    """
    spectral_embedding expects an affinity matrix that already has the similarity kernel applied.
        We apply the exponent here to be consistent with the mapping from distances to fuzzy
        simplices (affinities) via -log
    """
    print("Computing adjacency_matrix with {} neighbors and {} metric".format(n_neighbors, metric))
    graph = get_adjacency_matrix(X=X, n_neighbors=n_neighbors, metric=metric)
    print("Computing spectral_embedding")
    return spectral_embedding(graph, n_components=2)
