"""
Run a standard analysis on the output of mcmc sampling.
Generate figures in a directory.

1. Load in trajectory 
2. Calculate Rhat
3. Check chain convergances by eye
4. Load in references
5. Calculate AUC for references

MCMCM Fields
- potential energy

Coordinates are transformed to an unconstrained space


"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
    init_to_uniform,
    )
import click
import pickle as pkl
from pathlib import Path
import time
from typing import NamedTuple
import math
import sys
import xarray as xr
import _model_variations
from numpyro.infer.util import (
    log_density
        )
import _BSASA_functions
import flyplot
import sklearn
import os

class MCMCPlotData(NamedTuple):
    xlabel: str
    ylabel: str
    fmt: str
    title: str
   
__default_mcmc_plot_data = MCMCPlotData(
        xlabel = 'step',
        ylabel = 'auc',
        fmt = 'k.',
        title = 'direct_benchmark')

def load(fpath):
    with open(fpath, 'rb') as f: dat = pkl.load(f)
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

def auc_from_samples(samples: dict, nsamples, auc_func = "model12"):
    """
    Params:
      samples : a dictionary keyed by site name 
      nsamples : int - number of samples
    """
    if auc_func == "model12":
        direct_matrix = load_direct_benchmark().reference.matrix
        auc_func = partial(auc_from_sample_model12, direct_matrix=direct_matrix) 
    auc_array = jnp.zeros(nsamples)
    for i in range(nsamples):
        param_dict = {key: samples[key][i, ...] for key in samples}
        auc = auc_func(param_dict)
        auc_array = auc_array.at[i].set(auc)
    return auc_array

def auc_from_sample_model12(param_dict, direct_matrix, nnodes=1879):
    """
    Params
      param_dict : keyed by sample site
      direct_matrix : reference xarray
      cos_sim_matrix : label xarray
      nnodes : number of nodes in network
    """
    # Truncate the matrices to the smaller one
    cos_sim_matrix = load_model12_cos_sim_matrix()
    a1, a2 = direct_matrix.shape
    b1, b2 = cos_sim_matrix.shape
    assert a1 == a2
    assert b1 == b2
    if a1 > b1:
        # Truncate direct to cos sim matrix
        direct_matrix = direct_matrix.sel(preyu=cos_sim_matrix.preyu, preyv=cos_sim_matrix.preyv)
    elif b1 > a1:
        cos_sim_matrix = cos_sim_matrix.sel(preyu=direct_matrix.preyu, preyv=direct_matrix.preyv)
    else:
        ...
    n_possible_edges = math.comb(nnodes, 2)
    weight_array = weights2xarray(param_dict['w'], nnodes, cos_sim_matrix)
    direct_matrix = direct_matrix.sel(preyu=weight_array.preyu, preyv=weight_array.preyv)
    direct_matrix.values = direct_matrix > 0
    n_edges = np.sum(np.tril(direct_matrix, k=-1))
    thresholds = np.arange(0, 1, 0.01)
    pps, tps = _BSASA_functions.pp_tp_from_pairwise_prediction_matrix_and_ref(
        direct_matrix,
        weight_array,
        thresholds)
    pps = np.array(pps)
    tps = np.array(tps)
    tpr = tps / n_edges
    ppr = pps / n_possible_edges
    auc = sklearn.metrics.auc(
        x = ppr,
        y = tpr,
        )
    return auc

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

def save_mcmc_auc_plot(auc_array):
    fig, ax = plt.subplots()

def save_simple_scatter(ydata, xlabel, ylabel, title, savename, scatter_kwargs = None, dpi=300):
    if scatter_kwargs is None:
        scatter_kwargs = {}
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(ydata)), np.array(ydata), **scatter_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(savename, dpi=dpi)


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

def get_auc_model14(traj):
    N_expected_nodes = 3005
    N_expected_edges = math.comb(N_expected_nodes, 2)
    tril_indices = np.tril_indices_from(N_expected_nodes, k=-1)

    # Are the nodes in the correct order?

    with open(traj, "rb") as f:
        traj = pkl.load(f)
    
    with open("direct_benchmark.pkl", "rb") as f:
        direct = pkl.load(f)
    
    with open("cocomplex_benchmark.pkl", "rb") as f:
        cocomplex = pkl.load(f)

    direct_reference = direct.reference.matrix.values[tril_indices]
    del direct
    cocomplex_reference = cocomplex.reference.matrix.values[tril_indices]
    del cocomplex
    assert direct_reference.shape == (N_expected_edges,)
    assert cocomplex_reference.shape == (N_expected_edges,)

    nsamples, nedges = traj['samles']['pT'].shape 

    for edge_list in traj['samples']['pT']:
        assert edge_list.shape == (N_expected_edges,) edge_list.shape
        
         

def model14_traj2analysis(traj: str):
    """
    traj : str to a trajectory pickle file. e.g, 0_model14_0.pkl
    
    Output:
      saves an analysis_file. 0_model14_0_analysis.pkl
    """
    direct, cocomplex = get_auc_model14(traj)
    
    analysis = {
                "direct_auc" : direct,    # Area under curve for each model
                "cocomplex_auc" : cocomplex, # Area under curve for each model
    }
    
    top_analysis = {"top_direct_auc" : max(direct),
                    "top_cocomplex_auc" : max(cocomplex),
                    "lowest_potential_energy" : min(traj['extra_fields']['potential_energy']),
    }




@click.command()
@click.option("--trajectory-name" ,help="prefix to the trajectory pickle file")
@click.option("--analysis-name" ,help="a name for the analysis")
@click.option("--model-name")
@click.option("--fig-dpi", type=int, default=300)
def main(trajectory_name, analysis_name, fig_dpi, model_name):
    dir_name = analysis_name + "__" + trajectory_name
    assert not Path(dir_name).is_dir()
    os.mkdir(dir_name) 

    with open(trajectory_name + ".pkl", "rb") as f:
        d = pkl.load(f)
    energy = np.array(d['extra_fields']['potential_energy']) 
    plt.hist(energy, bins=50)
    plt.xlabel("score")
    plt.ylabel("frequency")
    plt.savefig(dir_name + "/potential_energy_hist.png", dpi=fig_dpi)
    plt.close()

    x = np.arange(len(energy))
    plt.plot(x, energy, 'k.')
    plt.ylabel("score")
    plt.xlabel("step")
    plt.savefig(dir_name + "/caterpillar.png", dpi=fig_dpi)
    

if __name__ == "__main__":
    main()
