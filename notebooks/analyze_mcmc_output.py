"""
Run a standard analysis on the output of mcmc sampling.
Generate figures in a directory.

1. Load in trajectory 
2. Calculate Rhat
3. Check chain convergances by eye
4. Load in references
5. Calculate AUC for references
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import numpyro
import numpyro.distributions as dist
from functools import partial
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
    init_to_uniform,
    )
import click
import pickle as pkl
import time
import math
import sys
import xarray as xr
import _model_variations
from numpyro.infer.util import (
    log_density
        )

def load(fpath):
    with open(fpath, 'rb') as f:
        dat = pkl.load(f)
    return dat

def load_direct_benchmark(path="direct_benchmark.pkl"):
    return load(path)

def log_density_from_samples(model, samples: dict, nsamples: int, model_args = (None,), model_kwargs = None):
    """
    Params:
      model : Numpyro model
      samples : a dictionary keyed by site name 
      nsamples : int - number of samples
      model_args : tuple 
      model_kwargs : dict 
    """
    if model_kwargs is None:
        model_kwargs = {}
    log_prob_array = jnp.zeros(nsamples)
    for i in range(nsamples):
        param_dict = {key: samples[key][i, ...] for key in samples}
        lp_, _ = log_density(model, model_args, model_kwargs, param_dict) 
        log_prob_array = log_prob_array.at[i].set(lp_)
    return log_prob_array

def auc_from_samples(samples: dict, nsamples, auc_func):
    """
    Params:
      samples : a dictionary keyed by site name 
      nsamples : int - number of samples
    """
    auc_array = jnp.zeros(nsamples)
    for i in range(nsamples):
        param_dict = {key: samples[key][i, ...] for key in samples}
        auc = auc_func(param_dict)
        auc_array = auc_array.at[i].set(auc)
    return auc_array

def auc_from_sample(param_dict, reference):
    """
    
    
    """
    ...

def trajectory_data2data_array(trajectory, nnodes):
    """
    """
    nsamples, nedges = trajectory.shape   
    data_array_shape = (nsamples, nnodes, nnodes)
    data_array = jnp.zeros(data_array_shape)
    tril_a, tril_b = jnp.tril_indices(nnodes, k=-1) 
    for i in range(nsamples):
        data_array = data_array.at[i, tril_a, tril_b].set(trajectory[i, :]) 
        L = data_array[i, :, :]
        A = L + L.T
        data_array = data_array.at[i, :, :].set(A)
    return data_array

def calculate_auc(trajectory_data_array, reference_matrix):
    draws, nnodes1, nnodes2 = trajectory_data_array.shape
    auc_vector = jnp.zeros(draws) 

def get_trajectory_auc(edgelist_trajectory, direct_benchmark, selector):
    ...


def load_model12_cos_sim_matrix():
    """
    Load in the cos similarity matrix used in model12
    """
    with open("direct_benchmark.pkl", "rb") as f:
        cos_sim_matrix = pkl.load(f).prediction.cosine_similarity.matrix
    with open("tensor_saint_scores.pkl", "rb") as f:
        tensor_saint_scores = pkl.load(f)
    prey_sel = ((tensor_saint_scores > 0).sum(dim = ["bait", "condition"]) > 0)
    prey_sel = prey_sel.sortby("preyu")
    cos_sim_matrix = cos_sim_matrix.sortby("preyu") 
    cos_sim_matrix = cos_sim_matrix.sortby("preyv")
    prey_sel = prey_sel.sel(preyu=cos_sim_matrix.preyu)
    prey_selv = xr.DataArray(prey_sel.values, coords={"preyv": prey_sel.preyu.values})
    cos_sim_matrix = cos_sim_matrix.sel(preyu=prey_sel, preyv=prey_selv)
    return cos_sim_matrix

def weights2xarray(w, nnodes, cos_sim_matrix):
    """
    Converts the sample site 'w' to a pairwise matrix
    in xarray format. For use in conjuction with model12 from _model_variations.py
    """
    tril_indices = jnp.tril_indices(nnodes, k=-1)
    A = jnp.zeros((nnodes, nnodes))
    A = A.at[tril_indices].set(w)
    A = A + A.T  # non-negative weighted adjacency matrix with 0 diagonal.
    A = np.array(A)
    weight_data = xr.DataArray(A, coords=cos_sim_matrix.coords)
    return weight_data

if __name__ == "__main__":
    main()
