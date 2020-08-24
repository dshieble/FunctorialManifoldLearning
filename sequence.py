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


def mutate_sequence(sequence):
    new_sequence = [s for s in sequence]
    ix = np.random.randint(0, len(new_sequence))
    new_sequence[ix] = np.random.choice([x for x in ["A", "C", "G", "T"] if not x == sequence[ix]])
    return new_sequence


def generate_mutation_chain(base_sequence, num_mutations):
    current_sequence = base_sequence
    chain = []
    for i in range(num_mutations):
        current_sequence = mutate_sequence(current_sequence)
        chain.append(current_sequence)
    return chain


def sequence_distance(s1, s2):
    return np.sum(s1 != s2) / len(s1)


def evaluate_embeddings(embedding_sequence, num_mutations):
    base_embeddings = embedding_sequence[::num_mutations]
    final_embeddings = embedding_sequence[num_mutations-1:][::num_mutations]
    _, indices = NearestNeighbors(n_neighbors=1).fit(base_embeddings).kneighbors(final_embeddings)
    accuracy = sum(indices.ravel() == np.arange(len(indices))) / len(indices)
    return accuracy