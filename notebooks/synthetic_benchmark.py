"""
Create a named analysis folder
- network_data.pkl
- trajectory_i.pkl
- model_accuracy.json
  - { trajectory_id : [...]}
- model_score.json
  - { trajectory_id : [...]}

- model_top_accuracy.json
  {trajectory_id : top_accuracy}

- model_top_score.json
  {trajectory_id : top_score}

trajectory_groups.json
 - {group_id : [1],
    group_id : [457, 9}

- trajectory_group_top_accuracy.json 
  - {trajectory_id : [top_accuracy]}
- trajectory_group_top_score.json 
  - {trajectory_id : [top_score]}



Network Generating Function
- Give N nodes and an edge frequency of pi_T, generate a PPI network.
1. Generate edges at a frequency of pi_T
2. Select M bait.
3. Calculate number of paths (i, j) all pairs of distance <= max_path_length
4. Remove all edges not connected to one of the bait
5. Calculate pi_obs (the observed edge frequency
6. If a pair of proteins do not interact, generate AP-MS pearson R from the null distribution
7. If a pair of proteins interact generate AP-MS pearson R from the causal distribution
8. Gather into a similarity matrix
9. Return the true_network, bait_indices, and pearson R similarity matrix.

Visualize the Synthetic Data
- Observe the distribution

Set Priors
- Set common sense priors on pi_obs, mu, and sigma

Run A MCMC chains for B steps
- Initialize different starting positions
- Index trajectories
- Save model scores

Calculate the top scoring model and most accurate model for each trajectory
- O(A x B) 
- Build the score dictionary

Plot accuracy score correlation for a representative subset
- Select V accuracy, score pairs 
- Calculate pearson R

Define trajectory groups and save

Index combinations of trajectories from 1 to A
- For each trajectory group calculate the top scoring model and most accurate model
- Calculate combinations without replacement 

Naming Conventions
"""

import jax
import jax.numpy as jnp
import numpy as np
import math
import _model_variations as mv
import pickle as pkl
import xarray as xr

_test_key = jax.random.PRNGKey(0)

def undirected_weighted_adjacency_rng(key, N, edge_prob, Adtype=jnp.float32):
    """
    key : a jax random prng key
    """
    A = jax.random.bernoulli(key, edge_prob, shape=(N, N)).astype(Adtype)
    A = jnp.tril(A, k=-1)
    A = A + A.T
    A = A.at[jnp.diag_indices(N)].set(0)
    return A

def adjacency_and_distance_rng(key, N: int, edge_prob: float, max_path_length: int):
    """
    key : jax rng key
    N : number of nodes
    edge_prob: proportion of edges
    n_bait : int, the number of bait proteins
    max_path_length: int, the maximal path length to calculate in the distance matrix 
    """
    assert max_path_length > 2
    key, k2 = jax.random.split(key)
    A = undirected_weighted_adjacency_rng(key, N, edge_prob) 
    # Shortest paths up to 10
    f = jax.tree_util.Partial(mv.shortest_paths_up_to_N, N=max_path_length)
    f = jax.jit(f)
    D = f(A)
    diag_indices = jnp.diag_indices(N)
    #assert jnp.sum(A[diag_indices]) == 0, A[diag_indices]
    # Assign the self distances to 0
    D = D.at[diag_indices].set(0)
    assert jnp.alltrue(A == A.T), A
    return A, D 

def bait_list_rng(key, N, n_bait):
    """
    Given a number of nodes in the network N and the number of bait,
    randomly generate bait indices.
    """
    bait_indices = jax.random.choice(key, N, shape=(n_bait,), replace=False)
    return bait_indices

def jax_pairwise_matrix_2_xarray(M, row_dim="preyu", col_dim="preyv"):
    """
    Converts a 2 dimensional jax or numpy matrix to a DataArray
    with named coordinates
    """
    N, _ = M.shape
    M = xr.DataArray(M, coords = {row_dim : jnp.arange(N),
                                  col_dim : jnp.arange(N)})
    return M


def get_connected_network_from_distance(A, D, d_crit, bait_idx_array,
                                        convert_to_xr=True):
    """
    Given a distance matrix D and a list of bait proteins baits:
    remove all prey that are not within a distance d_crit to any bait protein 
    """
    # Function expects self distances to be 0, otherwise a bait would be removed from
    # itself
    N, _ = A.shape
    if convert_to_xr:
        A = jax_pairwise_matrix_2_xarray(A)
        D = jax_pairwise_matrix_2_xarray(D)
    diag_indices = jnp.diag_indices(N)
    assert jnp.sum(D.values[diag_indices]) == 0
    # Select Prey distances
    B  = D.values[bait_idx_array, :]
    # Update the bait such that the distance to self is 0
    mask = B <= d_crit 
    # Take the sum of the columns such that if any prey has a distance le d_crit
    # The prey is kept in the network
    prey_idx = np.sum(mask, axis=0) > 0
    Aconnected = A.sel(preyu=prey_idx, preyv=prey_idx)
    return Aconnected, prey_idx

def get_bait_prey_network(rng_key, n_prey, n_bait: int, d_crit,
                          edge_prob=0.2, max_path_length=21):
    """
    A jax PRNGKey
    n_bait : the number of unique bait types
    d_crit : The maximal distance to at which prey are not considered connected
    to the bait
    edge_prob : the independant edge probability
    max_path_length : the maximal length of paths for distance matrix calculation
    """
    assert n_prey > n_bait , "There must be more prey types than bait types"
    keys = jax.random.split(rng_key, 2) 
    A, D = adjacency_and_distance_rng(keys[0], n_prey, edge_prob, max_path_length)  
    bait_idx_array = bait_list_rng(keys[1], n_prey, n_bait)
    Aconnected, prey_idx = get_connected_network_from_distance(A, D, d_crit,
                           bait_idx_array)
    return Aconnected, bait_idx_array, prey_idx
     



def data_from_network_model14_rng(rng_key, A, mu=0.23, sigma=0.21, ntest=3005):
    """
    Assume a binary matrix input A 
    Given an adjacency matrix  
    mu : 0.23
    sigma : 0.21
    Assumptions : Same number of conditions
    Params:
    1. Load in the network
    2. Count the number of edges
    3. Generate that many samples from the causal distribution
    4. Generate the remaining using the null 
    """
    k1, k2 = jax.random.split(rng_key)
    A = A.astype(bool)
    causal = jax.random.normal(k1, A.shape) * sigma + mu  
    # Load in null data
    with open("shuffled_apms_correlation_matrix.pkl", "rb") as f:
        shuffled_apms_correlation_matrix = pkl.load(f)
    shuffled_apms_correlation_matrix = shuffled_apms_correlation_matrix[0:ntest, 0:ntest]
    flattened_shuffled_apms = mv.matrix2flat(
            jnp.array(shuffled_apms_correlation_matrix, dtype=jnp.float32))
    null_dist = mv.Histogram(flattened_shuffled_apms, bins=1000)
    Null = null_dist.sample(k2, sample_shape=A.shape)
    Samples = jnp.where(A, causal, Null) 
    Samples = jnp.tril(Samples, k=-1)
    Samples = Samples + Samples.T
    return Samples


