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
from sklearn.neighbors import NearestNeighbors

NUCLEOTIDE_LIST = ["A", "C", "G", "T"]

def generate_dna_sequences(num_mutations, num_starts, num_modifications=100, sequence_length=1000):
    """
    Generate the DNA Simulation Dataset
    Args:
        num_mutations: The number of mutations to apply to each DNA string
        num_starts: The number of sequences to mutate
        num_modifications: The number of modifications to apply when creating the starts
        sequence_length: The length of each DNA sequence
    Returns
        Tuple of (list of base sequence ids for each sequence, condensed_distance_metric)
    """
    base_sequence_list = []
    base = np.random.choice(NUCLEOTIDE_LIST, sequence_length)
    for i in tqdm(range(num_starts)):
        modified_sequence = [s for s in base]
        for i in range(num_modifications):
            modified_sequence = mutate_sequence(modified_sequence)
        base_sequence_list.append(np.array(modified_sequence)[None,...])
    base_sequences = np.vstack(base_sequence_list)

    base_sequence_ids = []
    sequences = []
    for i, b in tqdm(enumerate(base_sequences)):
        base_sequence_ids += [i for j in range(num_mutations)]
        sequences += generate_mutation_chain(base_sequence=b, num_mutations=num_mutations)

    return base_sequence_ids, sequences


def mutate_sequence(sequence):
    """
    Apply a mutation step to the DNA sequence
    """
    new_sequence = [s for s in sequence]
    ix = np.random.randint(0, len(new_sequence))
    new_sequence[ix] = np.random.choice([x for x in NUCLEOTIDE_LIST if not x == sequence[ix]])
    return new_sequence


def generate_mutation_chain(base_sequence, num_mutations):
    """
    Apply a series of mutations to the base_sequence to generate a chain of mutations
    """
    current_sequence = base_sequence
    chain = []
    for i in range(num_mutations):
        current_sequence = mutate_sequence(current_sequence)
        chain.append(current_sequence)
    return chain


def sequence_distance(s1, s2):
    """
    Compute the distance between DNA sequences
    """
    return np.sum(s1 != s2) / len(s1)



def evaluate_embeddings(embedding_sequence, num_mutations, boostrap_size=100, num_bootstraps=100):
    """
    Evaluate the embeddings by computing the number of embedding sequences for which the nearest neighbor of
       the final embedding among all base embeddings is the base embedding of that sequence
    Args:
        embedding_sequence: A list of embeddings in which every block of `num_mutations` embeddings corresponds to
            one embedding that has been transformed repeatedly as `base_embedding,...,final_embedding`
        num_mutations: The number of mutations that had been applied to each embedding
    Return:
        (accuracy, standard_error)
    """

    base_embeddings = embedding_sequence[::num_mutations]
    final_embeddings = embedding_sequence[num_mutations-1:][::num_mutations]
    _, nn_indices = NearestNeighbors(n_neighbors=1).fit(base_embeddings).kneighbors(final_embeddings)
    accuracy = sum(nn_indices.ravel() == np.arange(len(nn_indices))) / len(nn_indices)
    
    # Accuracy is a proportion, so we can use closed form standard error
    accuracy_sem = np.sqrt(accuracy*(1-accuracy) / len(nn_indices))
    return accuracy, accuracy_sem
