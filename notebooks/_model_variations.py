"""
An organized approach to scoring and sampling

Scoring Functions to test
Availible:
- calculate path length (path term)
- pairwise term 

Terms:
- cosine similarity term (CS)
- path-length term (PL)
- CS & PL 
- GI pairwise term
- all three


Sampling:

- 4 chains
- 5000 draws per chain
- Save the warmup perdiod

- 1 job per chain
SampleBatch: 
- A collection of several jobs and chains grouped together

Composite connectivity

1. Set up a particle system with 1 bead per node
2. Define composites
3. Apply composite connectivity
4. Calculate contact frequencies
5. Calculate a network based on composite connectivity
6. Compare
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import Partial
import numpy as np
import numpyro
import numpyro.distributions as dist
from functools import partial
from itertools import combinations
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
    init_to_uniform,
    )
from numpyro.contrib.funsor import config_enumerate
from numpyro.util import format_shapes
import click
from pathlib import Path
import pandas as pd
import pickle as pkl
import time
import math
import sys
import xarray as xr
import logging
import json
from typing import Any, NamedTuple
import pdb

Array = Any

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs) 

def basic_bernoulli_adjacency1(
        spectral_count):
    n_prey = 3_062
#    assert len(spectral_count.preyu) == n_prey
    shape = (n_prey, n_prey)
    A = jnp.ones(shape, dtype=bool)
    probs = A * 0.01 
    probs = probs * (jnp.eye(n_prey) == 0)
#    A = numpyro.sample('A', dist.Normal(probs))
    A = numpyro.sample('A', dist.RelaxedBernoulli(probs,
        temperature=0.5))
#    L = jnp.tril(A, k=-1)
#    A = A + A.T

def relaxed_bernoulli1(x):
    n_prey = 36
    event_shape = (n_prey, n_prey)
    A = jnp.ones(event_shape, dtype=bool)
    probs = A * 0.01 
    probs = probs * (jnp.eye(n_prey) == 0)
    A = numpyro.sample('A', dist.RelaxedBernoulli(
        temperature=0.5, probs=probs))
    L = jnp.tril(A, k=-1)
    A = L + L.T

def model(y):  # y has dimension N x d
#    d = y.shape[1]
#    N = y.shape[0]
    N = 4
    d = 3_062
    y = jnp.ones((4, d))
    # Vector of variances for each of the d variables
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    # Lower cholesky factor of a correlation matrix
    concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(d, concentration))
    # Lower cholesky factor of the covariance matrix
    sigma = jnp.sqrt(theta)
    # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
    L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)
    # Vector of expectations
    mu = jnp.zeros(d)
    with numpyro.plate("observations", N):
        obs = numpyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y)
    return obs

def model(y):
    N = 4
    d = 3_062
    y = jnp.ones((N, d))
    K_corr = numpyro.sample('K', dist.LKJ(d, 1.5))
    mu = jnp.zeros(d)
    with numpyro.plate("observations", N):
        obs = numpyro.sample("obs", dist.MultivariateNormal(mu, precision_matrix=K), obs=y)
    return obs

def model(y):
    N = 4
    d = 30 #3_062
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    y = jnp.ones((N, d))
    U = numpyro.sample('U', dist.LKJ(d, 1.5))
    mu = jnp.zeros(d)
    obs = numpyro.sample("obs", dist.MultivariateNormal(
        mu, precision_matrix=U), obs=y)

def model(y):  # y has dimension N x d
    N = 4
    d = 10 
    y = jnp.ones((N, d))
    y1 = y[0, :].reshape((d, 1))
    y2 = y[1, :].reshape((d, 1))
    y3 = y[2, :].reshape((d, 1))
    y4 = y[3, :].reshape((d, 1))
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
    corr_mat = numpyro.sample("corr_mat", dist.LKJ(d, concentration))
    sigma = jnp.sqrt(theta)
    cov_mat = jnp.outer(theta, theta) * corr_mat
    cov_mat = jnp.tril(cov_mat) + jnp.tril(cov_mat, k=-1).T 
    mu = jnp.zeros(d)
    obs1 = numpyro.sample("obs1", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y1)
    obs2 = numpyro.sample("obs2", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y2)
    obs3 = numpyro.sample("obs3", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y3)
    obs4 = numpyro.sample("obs4", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y4)

def model(y): # y has dimension N x d
    d = y.shape[1]
    print(f"y: {y.shape}")
    N = y.shape[0]
    # Vector of variances for each of the d variables
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    print(f"theta: {theta.shape}")
    concentration = jnp.ones(1) # Implies a uniform distribution over
    prior = dist.LKJ(d, concentration)
    print(f"batch: {prior.batch_shape}")
    print(f"event {prior.event_shape}")
    #print(f"sample: {prior.sample_shape}")
    corr_mat = numpyro.sample("corr_mat", prior)
    assert jnp.all(corr_mat[0, :, :] == cor_mat[0, :, :])
    print(f"corr_mat: {corr_mat.shape}")
    print(f"corr_mat info {type(corr_mat)}")
    sigma = jnp.sqrt(theta)
    # we can also use a faster formula `cov_mat = jnp.outer(theta, theta) * corr_
    cov_mat = jnp.matmul(jnp.matmul(jnp.diag(sigma), corr_mat), jnp.diag(sigma))
    print(f"cov_mat: cov_mat.shape")
    #L = jnp.tril(cov_mat)
    #cov_mat = L + jnp.tril(L, k=-1).T
    # Vector of expectations
    mu = jnp.zeros(d)
    print(f"mu: {mu.shape}")
    with numpyro.plate("observations", N):
        obs = numpyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=cov_mat), obs=y)
        return obs

#assert jnp.all(cov_mat == cov_mat.T), jnp.where(
#    cov_mat != cov_mat.T)
def eight_schools(J, sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

def cov_model1(dim):
    Sigma = numpyro.sample("s", dist.LKJCholesky(dim, 1.1))
    print(Sigma)
#    numpyro.sample("obs", dist.MultivariateNormal(jnp.arange(dim), scale_tril=Sigma),
#                   obs=jnp.arange(dim))
#    return Sigma

def cov_model2(dim):
    """
    Model works
    """
    mu = jnp.zeros(dim)
    sigma = jnp.eye(dim)
    numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, covariance_matrix=sigma))


def cov_model3(dim):
    """
    """
    #mu = jnp.zeros(dim)
    sigma = numpyro.sample("sigma", dist.LKJCholesky(dim, concentration=jnp.array(1.5, dtype=jnp.float64)))
    #numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, covariance_matrix=sigma))

def get_cov_model4_init_strategy(dim):
    values = {"sigma": jnp.eye(dim),
              "mvn": jnp.zeros(dim)}
    return partial(init_to_value, values=values)

def cov_model4(dim):
    """
    """
    mu = jnp.zeros(dim)
    sigma = numpyro.sample("sigma", dist.LKJCholesky(dim, concentration=jnp.array(1.5, dtype=jnp.float64)))
    #This is wierdly misspecified
    numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, covariance_matrix=sigma))

def cov_model5(dim):
    """
    Use model4 init strategy
    """
    mu = jnp.zeros(dim)
    sigma = numpyro.sample("sigma", dist.LKJCholesky(dim, concentration=1.5))
    #This is wierdly misspecified
    numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, scale_tril=sigma), obs=jnp.zeros(dim))

def cov_model6(dim):
    """
    Use model4 init strategy
    """
    _mu_hyper = jnp.ones(dim)
    mu = numpyro.sample("mu", dist.Poisson(_mu_hyper)) 
    sigma = numpyro.sample("sigma", dist.LKJCholesky(dim, concentration=1.5))
    #This is wierdly misspecified
    numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, scale_tril=sigma), obs=jnp.zeros(dim))

def get_cov_model7_init_strategy(dim):
    values = {"sigma1": jnp.eye(dim),
        "sigma2": jnp.eye(dim),
        "sigma3": jnp.eye(dim)}
    return partial(init_to_value, values=values)

def cov_model7(dim):
    """
    Use model4 init strategy
    """
    mu = jnp.zeros(dim)
    sigma1 = numpyro.sample("sigma1", dist.LKJCholesky(dim, concentration=1.5))
    sigma2 = numpyro.sample("sigma2", dist.LKJCholesky(dim, concentration=1.5))
    sigma3 = numpyro.sample("sigma3", dist.LKJCholesky(dim, concentration=1.5))
    #This is wierdly misspecified
    numpyro.sample("mvn1", dist.MultivariateNormal(loc=mu, scale_tril=sigma1), obs=jnp.zeros(dim))
    numpyro.sample("mvn2", dist.MultivariateNormal(loc=mu, scale_tril=sigma2), obs=jnp.zeros(dim))
    numpyro.sample("mvn3", dist.MultivariateNormal(loc=mu, scale_tril=sigma3), obs=jnp.zeros(dim))

def cov_model8(dim):
    _mu_hyper = jnp.ones(dim)
    # Vector of data means
    mu = numpyro.sample("mu", dist.HalfNormal(_mu_hyper)) 
    # Vector of variances for each of the d variables
    theta = numpyro.sample("theta", dist.HalfCauchy(mu))
    # Lower cholesky factor of a correlation matrix
    # concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(dim, concentration=1.5))
    # Lower cholesky factor of the covariance matrix
    sigma = jnp.sqrt(theta)
    # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
    L_Omega = sigma[..., None] * L_omega
    #L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)
    # Vector of expectations
    numpyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=jnp.zeros(dim))

def get_cov_model9_init_strategy(dim):
    values = {"L_omega": jnp.eye(dim)}
    return partial(init_to_value, values=values)

def cov_model9(dim):
    """
    Use model4 init strategy
    Does the model work with precision matrix parameterization?
    """
    _mu_hyper = jnp.ones(dim)
    # Vector of data means
    mu = numpyro.sample("mu", dist.HalfNormal(_mu_hyper)) 
    # Vector of variances for each of the d variables
    theta = numpyro.sample("theta", dist.HalfCauchy(mu))
    # Lower cholesky factor of a correlation matrix
    # concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(dim, concentration=1.5))
    # Lower cholesky factor of the covariance matrix
    sigma = jnp.sqrt(theta)
    # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
    L_Omega = sigma[..., None] * L_omega
    #L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)
    # Vector of expectations
    Omega = L_Omega @ L_Omega.T
    numpyro.sample("obs", dist.MultivariateNormal(mu, precision_matrix=Omega), obs=jnp.zeros(dim))

def model_10_wt(dim, condition_sel="wt"):
    #Data prep
    bait_sel = ["CBFB", "ELOB", "CUL5"] # Remove LRR1
    spectral_count_xarray = load("spectral_count_xarray.pkl")
    spectral_count_xarray = spectral_count_xarray.sel(
            condition=condition_sel, bait=bait_sel)
    prey_isel = np.arange(dim)
    spectral_count_xarray = spectral_count_xarray.isel(
            preyu=prey_isel) 
    y = spectral_count_xarray.sel(AP=True) - spectral_count_xarray.sel(AP=False)
    y = y.transpose('preyu', 'bait', 'rep')
    y = y.values
    #Model 
    L_omega = numpyro.sample("L_omega", dist.LKJ(dim, concentration=1.0)) 
    mu = numpyro.sample("mu", dist.HalfNormal(_mu_hyper))
    sigma = jnp.sqrt(mu)
    L_Omega = sigma[..., None] * L_omega 
    with numpyro.plate('rep', 4):
        with numpyro.plate('bait', 3):
            dist.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_omega), obs=y) 

def _body_sum_of_path_weights_up_to_N(i, v):
    A, AN, W = v
    # Update the distance matrix
    W = W + AN 
    # 
    AN = A @ AN
    v = A, AN, W
    return v


def sum_of_path_weights_up_to_N(A, N):
    W = A 
    A2 = A @ A
    A, AN, W = jax.lax.fori_loop(2, N, _body_sum_of_path_weights_up_to_N, (A, A2, W))
    return W 

def _body_shortest_paths_up_to_N(i, v):
    A, AN, D = v
    # Update the distance matrix
    D = jnp.where((AN !=0) & (D > i), i, D)
    # 
    AN = A @ AN
    v = A, AN, D
    return v


def shortest_paths_up_to_N(A, N):
    """
    Given an adjacency matrix A, calculate the distance matrix D
    of shortest paths, up to N.
    """
    D = jnp.where(A != 0, A, N)
    A2 = A @ A
    A, AN, D = jax.lax.fori_loop(2, N, _body_shortest_paths_up_to_N, (A, A2, D))
    return D 

shortest_paths_up_to_23 = Partial(shortest_paths_up_to_N, N=23)

def model_11_path_length(dim):
    # Input Information
    max_path_length = 23
    # Read in spectral count data
    with open("spectral_count_xarray.pkl", "rb") as f:
        data = pkl.load(f)
    with open("tensor_saint_scores.pkl", "rb") as f:
        tensor_saint_scores = pkl.load(f)
    #Exclude prey where the Saint score is zero
    prey_sel = ((tensor_saint_scores > 0).sum(dim = ["bait", "condition"]) > 0)
    prey_sel = prey_sel.sortby("preyu")
    data = data.sortby("preyu")
    assert np.all(data.preyu.values == prey_sel.preyu.values)
    # Select data where saint > 0
    spectral_count_xarray_sel = data.sel(preyu=prey_sel)
    tensor_saint_scores = tensor_saint_scores.sel(preyu=prey_sel)
    # Change input information to jax arrays
    saint_elob_wt = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="wt"))
    saint_elob_vif = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="vif"))
    saint_elob_mock = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="mock"))

    saint_cbfb_wt = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="wt"))
    saint_cbfb_vif = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="vif"))
    saint_cbfb_mock = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="mock"))

    # Model representation
    # The dimension is determined by kj
    #The prior edge weight is goverened by Cosine Similarity
    m_possible_edges = int(math.comb(dim, 2))
    tril_indices = jnp.tril_indices_from(A, k=-1)
    # Beta distribution over edge weights 
    alpha = jnp.ones(m_possible_edges) * 0.408
    beta = jnp.ones(m_possible_edges) * 0.116
    #Beta distributions ensures no negative cycles
    edge_weight_list = numpyro.sample("w", dist.Beta(alpha, beta))

    A = jnp.zeros((N, N))
    A = A.at[tril_indices].set(edge_weight_list)
    A = jnp.where(A >= 0.5, 1., 0.)
    #Shortest Path Matrix

    D = shortest_paths_up_to_23(A)

    P = D < max_path_length 
    P = P.at[DIAG_INDICES].set(1) # Set self paths to 1 because bait are in their own
    # Scoring - path presence
    

    # p(SaintScore | D)

    ##Path presence
    ## pulldown with probability 1.
    ##Score based on saint scores
    #ELOB_paths = P[ELOB_index, :]
    #CUL5_paths = P[CUL5_index, :]
    #CBFB_paths = P[CBFB_index, :]
    #numpyro.sample("ELOB_wt", dist.Normal(ELOB_paths, sigma),   obs=ELOB_wt)
    #numpyro.sample("ELOB_vif", dist.Normal(ELOB_paths, sigma),  obs=ELOB_vif)
    #numpyro.sample("ELOB_mock", dist.Normal(ELOB_paths, sigma), obs=ELOB_mock)
    #numpyro.sample("CUL5_wt", dist.Normal(CUL5_paths, sigma),   obs=CUL5_wt)
    #numpyro.sample("CUL5_vif", dist.Normal(CUL5_paths, sigma),  obs=CUL5_vif)
    #numpyro.sample("CUL5_mock", dist.Normal(CUL5_paths, sigma), obs=CUL5_mock)


model_10_vif = partial(model_10_wt, condition_sel="vif")
model_10_mock = partial(model_10_wt, condition_sel="mock")
model_10_wt_vif = partial(model_10_wt, condtion_sel=["wt", "vif"])
model_10_wt_mock = partial(model_10_wt, condition_sel=["wt", "mock"])
model_10_vif_mock = partial(model_10_wt, condition_sel=["vif", "mock"])
model_10_wt_vif_mock = partial(model_10_wt,
    condition_sel=["wt", "vif", "mock"])


def model12(model_data):
    """
    Params: model_data
      model_data is a place holder and passes no information
    """
    ## Input information
    max_path_length = 23
    # Read in spectral count data
    #with open("spectral_count_xarray.pkl", "rb") as f:
    #    data = pkl.load(f)
    with open("direct_benchmark.pkl", "rb") as f:
        cos_sim_matrix = pkl.load(f).prediction.cosine_similarity.matrix
    with open("tensor_saint_scores.pkl", "rb") as f:
        tensor_saint_scores = pkl.load(f)
    #with open("df_new.pkl", "rb") as f:
    #    df_new = pkl.load(f)
    #name2uid = {r.name: r['Prey'] for i,r in df_new.iterrows()}
    #del df_new
    ## Indices and Uniprot IDS
    #elob_uid = name2uid['ELOB']
    #cbfb_uid = name2uid['PEBB']
    #cul5_uid = name2uid['CUL5']
    #Exclude prey where the Saint score is zero
    #This filter results in 1879 unique prey types
    prey_sel = ((tensor_saint_scores > 0).sum(dim = ["bait", "condition"]) > 0)
    prey_sel = prey_sel.sortby("preyu")
    #Change to 500 nodes
    tensor_saint_scores = tensor_saint_scores.sortby("preyu")
    cos_sim_matrix = cos_sim_matrix.sortby("preyu") 
    cos_sim_matrix = cos_sim_matrix.sortby("preyv")
    # Filter by prey_sel
    tensor_saint_scores = tensor_saint_scores.sel(preyu=prey_sel)
    # Reduce prey_sel to cos sim matrix index
    prey_sel = prey_sel.sel(preyu=cos_sim_matrix.preyu)
    prey_selv = xr.DataArray(prey_sel.values, coords={"preyv": prey_sel.preyu.values})
    cos_sim_matrix = cos_sim_matrix.sel(preyu=prey_sel, preyv=prey_selv)
    # Reduce to 500 nodes
    #cos_sim_matrix = cos_sim_matrix.isel(preyu=i_sel, preyv=i_sel)
    #tensor_saint_scores = tensor_saint_scores.isel(preyu=i_sel)
    # Change input information to jax arrays
    saint_elob_wt = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="wt"))
    saint_elob_vif = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="vif"))
    saint_elob_mock = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="mock"))
    #  
    saint_cbfb_wt = jnp.array(tensor_saint_scores.sel(bait="CBFB", condition="wt"))
    saint_cbfb_vif = jnp.array(tensor_saint_scores.sel(bait="CBFB", condition="vif"))
    saint_cbfb_mock = jnp.array(tensor_saint_scores.sel(bait="CBFB", condition="mock"))
    #
    saint_CUL5_wt = jnp.array(tensor_saint_scores.sel(bait="CUL5", condition="wt"))
    saint_CUL5_vif = jnp.array(tensor_saint_scores.sel(bait="CUL5", condition="vif"))
    saint_CUL5_mock = jnp.array(tensor_saint_scores.sel(bait="CUL5", condition="mock"))
    assert np.all(tensor_saint_scores.preyu.values == cos_sim_matrix.preyu.values)
    #
    jax_cos_sim_matrix = jnp.array(cos_sim_matrix.values)
    # Model representation
    nnodes = len(saint_CUL5_wt) 
    tril_indices = jnp.tril_indices(nnodes, k=-1)
    diag_indices = np.diag_indices(nnodes)
    elob_idx = np.where(cos_sim_matrix.preyu == 'ELOB')[0].item()  
    cbfb_idx = np.where(cos_sim_matrix.preyu == 'PEBB')[0].item()
    cul5_idx = np.where(cos_sim_matrix.preyu == 'CUL5')[0].item()
    # truncate based on total number of nodes
    #The prior edge weight is goverened by Cosine Similarity
    m_possible_edges = int(math.comb(nnodes, 2))
    # Beta distribution over edge weights 
    alpha = jnp.ones(m_possible_edges) * 0.685 #0.408
    beta = jnp.ones(m_possible_edges) * 1.099   #0.116
    #Beta distributions ensures no negative cycles
    edge_weight_list = numpyro.sample("w", dist.Beta(alpha, beta))
    A = jnp.zeros((nnodes, nnodes))
    A = A.at[tril_indices].set(edge_weight_list)
    A = A + A.T  # non-negative weighted adjacency matrix with 0 diagonal.
    # p(cos_sim | A)
    STD2 = 0.2
    numpyro.sample("COS_SYM1", dist.Normal(A, STD2), obs=jax_cos_sim_matrix)
    # p(SAINT | D(A))
    # Discritize the matrix
    A = jnp.where(A >= 0.5, 1., 0.)
    #Shortest Path Matrix
    D = shortest_paths_up_to_23(A)
    P = D < max_path_length 
    P = P.at[diag_indices].set(1) # Set self paths to 1 because bait are in their own
    # p(ELOB | P(A))
    STD = 0.2
    P = jnp.array(P, dtype=jnp.int32)
    numpyro.sample("ELOB_WT", dist.Normal(P[elob_idx, :], STD), obs=saint_elob_wt)
    numpyro.sample("ELOB_VF", dist.Normal(P[elob_idx, :], STD), obs=saint_elob_vif)
    numpyro.sample("ELOB_MK", dist.Normal(P[elob_idx, :], STD), obs=saint_elob_mock)

    numpyro.sample("CBFB_WT", dist.Normal(P[cbfb_idx, :], STD), obs=saint_cbfb_wt)
    numpyro.sample("CBFB_VF", dist.Normal(P[cbfb_idx, :], STD), obs=saint_cbfb_vif)
    numpyro.sample("CBFB_MK", dist.Normal(P[cbfb_idx, :], STD), obs=saint_cbfb_mock)

    numpyro.sample("CUL5_WT", dist.Normal(P[cul5_idx, :], STD), obs=saint_CUL5_wt)
    numpyro.sample("CUL5_VF", dist.Normal(P[cul5_idx, :], STD), obs=saint_CUL5_vif)
    numpyro.sample("CUL5_MK", dist.Normal(P[cul5_idx, :], STD), obs=saint_CUL5_mock)
    # What to do about double counting the direct interactions?
    numpyro.sample("COS_SYM", dist.Normal(P < 3, STD), obs=jax_cos_sim_matrix)

def model13(model_data):
    """
    Params: model_data
      model_data is a place holder and passes no information


    Model:

      w ~ Beta(alpha, beta)  prior density of edges

    """
    ## Input information
    max_path_length = 23
    # Read in spectral count data
    #with open("spectral_count_xarray.pkl", "rb") as f:
    #    data = pkl.load(f)
    with open("direct_benchmark.pkl", "rb") as f:
        cos_sim_matrix = pkl.load(f).prediction.cosine_similarity.matrix
    with open("tensor_saint_scores.pkl", "rb") as f:
        tensor_saint_scores = pkl.load(f)
    #with open("df_new.pkl", "rb") as f:
    #    df_new = pkl.load(f)
    #name2uid = {r.name: r['Prey'] for i,r in df_new.iterrows()}
    #del df_new
    ## Indices and Uniprot IDS
    #elob_uid = name2uid['ELOB']
    #cbfb_uid = name2uid['PEBB']
    #cul5_uid = name2uid['CUL5']
    #Exclude prey where the Saint score is zero
    #This filter results in 1879 unique prey types
    prey_sel = ((tensor_saint_scores > 0).sum(dim = ["bait", "condition"]) > 0)
    prey_sel = prey_sel.sortby("preyu")
    #Change to 500 nodes
    tensor_saint_scores = tensor_saint_scores.sortby("preyu")
    cos_sim_matrix = cos_sim_matrix.sortby("preyu") 
    cos_sim_matrix = cos_sim_matrix.sortby("preyv")
    # Filter by prey_sel
    tensor_saint_scores = tensor_saint_scores.sel(preyu=prey_sel)
    # Reduce prey_sel to cos sim matrix index
    prey_sel = prey_sel.sel(preyu=cos_sim_matrix.preyu)
    prey_selv = xr.DataArray(prey_sel.values, coords={"preyv": prey_sel.preyu.values})
    cos_sim_matrix = cos_sim_matrix.sel(preyu=prey_sel, preyv=prey_selv)
    # Reduce to 500 nodes
    #cos_sim_matrix = cos_sim_matrix.isel(preyu=i_sel, preyv=i_sel)
    #tensor_saint_scores = tensor_saint_scores.isel(preyu=i_sel)
    # Change input information to jax arrays
    saint_elob_wt = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="wt"))
    saint_elob_vif = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="vif"))
    saint_elob_mock = jnp.array(tensor_saint_scores.sel(bait="ELOB", condition="mock"))
    #  
    saint_cbfb_wt = jnp.array(tensor_saint_scores.sel(bait="CBFB", condition="wt"))
    saint_cbfb_vif = jnp.array(tensor_saint_scores.sel(bait="CBFB", condition="vif"))
    saint_cbfb_mock = jnp.array(tensor_saint_scores.sel(bait="CBFB", condition="mock"))
    #
    saint_CUL5_wt = jnp.array(tensor_saint_scores.sel(bait="CUL5", condition="wt"))
    saint_CUL5_vif = jnp.array(tensor_saint_scores.sel(bait="CUL5", condition="vif"))
    saint_CUL5_mock = jnp.array(tensor_saint_scores.sel(bait="CUL5", condition="mock"))
    assert np.all(tensor_saint_scores.preyu.values == cos_sim_matrix.preyu.values)
    #
    jax_cos_sim_matrix = jnp.array(cos_sim_matrix.values)
    # Model representation
    nnodes = len(saint_CUL5_wt) 
    tril_indices = jnp.tril_indices(nnodes, k=-1)
    diag_indices = np.diag_indices(nnodes)
    elob_idx = np.where(cos_sim_matrix.preyu == 'ELOB')[0].item()  
    cbfb_idx = np.where(cos_sim_matrix.preyu == 'PEBB')[0].item()
    cul5_idx = np.where(cos_sim_matrix.preyu == 'CUL5')[0].item()
    # truncate based on total number of nodes
    #The prior edge weight is goverened by Cosine Similarity
    m_possible_edges = int(math.comb(nnodes, 2))
    # Beta distribution over edge weights 
    alpha = jnp.ones(m_possible_edges) * 0.685 #0.408
    beta = jnp.ones(m_possible_edges) * 1.099   #0.116
    #Beta distributions ensures no negative cycles
    edge_weight_list = numpyro.sample("w", dist.Beta(alpha, beta))
    A = jnp.zeros((nnodes, nnodes))
    A = A.at[tril_indices].set(edge_weight_list)
    A = A + A.T  # non-negative weighted adjacency matrix with 0 diagonal.
    # p(cos_sim | A)
    STD2 = 0.2
    numpyro.sample("COS_SYM1", dist.Normal(A, STD2), obs=jax_cos_sim_matrix)
    # p(SAINT | D(A))
    # Discritize the matrix
    A = jnp.where(A >= 0.5, 1., 0.)
    #Shortest Path Matrix
    #D = shortest_paths_up_to_23(A)
    #P = D < max_path_length 
    #P = P.at[diag_indices].set(1) # Set self paths to 1 because bait are in their own
    # p(ELOB | P(A))
    P = A @ A
    STD = 0.2
    #P = jnp.array(P, dtype=jnp.int32)
    numpyro.sample("ELOB_WT", dist.Normal(P[elob_idx, :], STD), obs=saint_elob_wt)
    numpyro.sample("ELOB_VF", dist.Normal(P[elob_idx, :], STD), obs=saint_elob_vif)
    numpyro.sample("ELOB_MK", dist.Normal(P[elob_idx, :], STD), obs=saint_elob_mock)

    numpyro.sample("CBFB_WT", dist.Normal(P[cbfb_idx, :], STD), obs=saint_cbfb_wt)
    numpyro.sample("CBFB_VF", dist.Normal(P[cbfb_idx, :], STD), obs=saint_cbfb_vif)
    numpyro.sample("CBFB_MK", dist.Normal(P[cbfb_idx, :], STD), obs=saint_cbfb_mock)

    numpyro.sample("CUL5_WT", dist.Normal(P[cul5_idx, :], STD), obs=saint_CUL5_wt)
    numpyro.sample("CUL5_VF", dist.Normal(P[cul5_idx, :], STD), obs=saint_CUL5_vif)
    numpyro.sample("CUL5_MK", dist.Normal(P[cul5_idx, :], STD), obs=saint_CUL5_mock)
    # What to do about double counting the direct interactions?
    numpyro.sample("COS_SYM", dist.Normal(P < 3, STD), obs=jax_cos_sim_matrix)

def BinomialApproximation(n, p):
    """
    The normal approximation to the Binomial distribution
    Valid for 
    """
    return dist.Normal(n * p, n * p * (1-p))

def f(n, p):
    return dist.Binomial

class CompositeConnectivity(dist.Distribution):
    """
    Given an edge array  and an edge_indexing_arr, place a lower bound on the
    sum of the edges at the indexes such that the sum is >= N - 1 where N is the length of the indexing list. 
    Params:
      edge_idx : an indexing array of edges in the composite
      N : the number of nodes in the composite
    """
    def __init__(self, edge_idx, N, scale=1.0, mu_w = 0.5, magnitude=1.0,validate_args=None):
        self.edge_idx = edge_idx
        self.N = N
        self.support = dist.constraints.real
        self.scale = scale
        self.n_choose_2 = math.comb(N, 2)
        self.mu_x = self.n_choose_2 * mu_w 
        self.mag = magnitude
        super(CompositeConnectivity, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
    def sample(self, key, sample_shape=()):
        ...
    def log_prob(self, value):
        edge_weights = value[self.edge_idx]
        x = jnp.sum(edge_weights) - self.N -1 -self.mu_x # composite connectivity 
        x = x * self.scale
        y = jax.nn.sigmoid(x+1) * self.mag # shift the curve to the right
        return y

def simple_bait_prey_score_fn(d, k=8, slope=1.0): 
    """
    This linear score maps to the exponential distribution
    The score is the average number of connected prey where connected is defined
    as the bait-prey distance less than k 
    Params:
      d : a bait-prey distance array
      k : int the distance thresholds
    """
    return jnp.mean(d < k) * slope

def simple_bait_prey_score_fn_dispatcher(
        edge_idxs,
        Ndense,
        maximal_shortest_path_to_calculate,
        binary_ppi_threshold,
        pc,
        bait_prey_slope):
    return Partial(simple_bait_prey_score_fn, k=maximal_shortest_path_to_calculate, slope=bait_prey_slope)

def sigmoid_bait_prey_score_fn(d, N, k=8, slope=10):
    """
    d : The bait prey distance vector (including the bait)
    N : the number of molecular types in the composite
    """
    x = jnp.sum(d < k) - N
    x = x + 0.5
    x = x * slope
    return (jax.nn.sigmoid(x) - 0.5) 

def bait_prey_connectivity_score_fn_dispatcher(
        edge_idxs,
        Ndense,
        maximal_shortest_path_to_calculate,
        binary_ppi_threshold,
        pc,
        bait_prey_slope):
    return Partial(bait_prey_connectivity_score_fn, k=maximal_shortest_path_to_calculate)

def bait_prey_connectivity_direct_score_fn(
        aij,
        composite_edge_ids,
        Nc,
        Nmax,
        k,
        max_paths,
        DTYPE=jnp.int32,
        loc=0,
        scale=1) -> float:
    """
    aij : the flattened edge array 
    Nc  : the number of proteins in the composite. Can be a random variable
    Nmax : Cannot be a random variable
    k   : the maximal bait prey distance at which a path is counted as disconnected
    max_paths : the maximal number of matrix multiplications to use in the Warshal algorithm
    DTYPE : don't change - the Adjacency matrix should be integer so that max paths is properly calculated
    """
    composite_edges = aij[composite_edge_ids]
    A = flat2matrix(composite_edges, Nmax)
    A = jnp.where(A < 0.5, jnp.array([0], dtype=DTYPE), jnp.array([1], dtype=DTYPE))
    D = shortest_paths_up_to_N(A, max_paths)
    distance2bait = D[0, :] # (N, )
    distance2bait = distance2bait.at[0].set(0) # set self distance to 0 
    n_bait_prey_interactions = jnp.sum(distance2bait < k)
    x = n_bait_prey_interactions - Nc
    return jax.lax.cond(x >= 0,
                        lambda x: jsp.stats.norm.logpdf(0, loc, scale),
                        lambda x: jsp.stats.norm.logpdf(x, loc, scale),
                        x)

def smart_binomial_distribution_dispatcher(n, p):
    if (n * p >= 5) and (n * (p-1)) >= 5:
        return BinomialApproximation(n, p)
    else:
        return dist.Binomial(n, p)

def bait_prey_score_norm_approx(
    composite_name,
    aij,
    composite_idxs,
    Ndense,
    Nmax,
    k,
    pc,
    max_distance = 22):
    """
    A component of probablistic model using Numpyro primitives
    """
    edge_idxs = jnp.array(node_list2edge_list(composite_idxs, Nmax))
    n_prey = numpyro.sample(composite_name + "_N", BinomialApproximation(Ndense, pc))
    score = bait_prey_connectivity_direct_score_fn(aij = aij,
       composite_edge_ids=edge_idxs,
       Nc = n_prey,
       Nmax = Ndense,
       k = k,
       max_paths = max_distance)
    numpyro.factor(composite_name + "_score", score)

def bait_prey_score_p_1(
    composite_name,
    aij,
    composite_idxs,
    Ndense,
    Nmax,
    k,
    pc,
    max_distance = 22):
    """
    A component of probablistic model using Numpyro primitives
    """
    edge_idxs = jnp.array(node_list2edge_list(composite_idxs, Nmax))
    #n_prey = numpyro.sample(composite_name + "_N", BinomialApproximation(Ndense, pc))
    n_prey = Ndense 
    score = bait_prey_connectivity_direct_score_fn(aij = aij,
        composite_edge_ids=edge_idxs,
        Nc = n_prey,
        Nmax = Ndense,
        k = k,
        max_paths = max_distance)
    numpyro.factor(composite_name + "_score", score)

def kth_bait_prey_score_p_is_1(
    aij, i, cd, Nmax, k=8, max_distance=22): 
    c_i = cd[i]
    c_i_nodes = c_i['nodes']
    c_i_N = c_i['N']
    c_i_t = c_i['t']
    bait_prey_score_p_1(composite_name=f"c{i}",
                    aij=aij,
                    composite_idxs=c_i_nodes,
                    Ndense=c_i_N,
                    Nmax=Nmax,
                    k=k,
                    pc=c_i_t,
                    max_distance=max_distance) 


def kth_bait_prey_score_norm_approx(aij, i, cd, Nmax, k=8, max_distance=22): 
    c_i = cd[i]
    c_i_nodes = c_i['nodes']
    c_i_N = c_i['N']
    c_i_t = c_i['t']
    bait_prey_score_norm_approx(composite_name=f"c{i}",
                    aij=aij,
                    composite_idxs=c_i_nodes,
                    Ndense=c_i_N,
                    Nmax=Nmax,
                    k=k,
                    pc=c_i_t,
                    max_distance=max_distance) 


def bait_prey_connectivity_score_fn(d, N, k=8, loc=0, scale=1):
    """
    d : An array of distances to the bait where the first distance (bait 2 bait) is 0
    N : The number of proteins in the composite including bait
    k : A distance over which two proteins are considered disconnected
    pc: The prior probability of a composite
    Bait Prey connectivity implies that there must be at least N bait prey paths (including bait to itself).
    
    pc implies that each protein type has a pc chance of being present in the composite.
       Therefore, Bait prey connectivity implies there must be at least N * pc bait prey paths to satisfy the restraint.
    
    """
    n_bait_prey_interactions = jnp.sum(d < k)
    x = n_bait_prey_interactions - N  # Satisfied >= 0
    return jax.lax.cond(x >= 0,
                        lambda x: jsp.stats.norm.logpdf(0, loc, scale),
                        lambda x: jsp.stats.norm.logpdf(x, loc, scale),
                        x)

vbait_prey_connectivity = jax.vmap(bait_prey_connectivity_score_fn)


def m_from_n_choose2(c, DTYPE=jnp.int32):
    """
    The inverse function of N choose two over valid values of n choose 2 
    """
    return jnp.ceil(jnp.sqrt(8 * c + 1) // 2 + 0.5).astype(DTYPE) 

def a_b(N: int, i: int) -> int:
    """
    Return the number of elements below and including the ith row. 
    """
    w = N - i
    return (w**2-w) // 2

def a_total(N: int) -> int:
    """
    Return the N choose two elements as the area of a triangle
    excluding the diagonal.
    """
    return (N**2 - N) // 2

def u(i: int, j: int, N: int) -> int:
    """
    Given indices i and j number of nodes N, return the
    edge index u
    """
    return a_total(N) - a_b(N, i) + j - i -1 

def i_from_u(u, N):
    """
    An iterative algorithm to find, given the value of the edge index
    u and the number of nodes N, find i.
    """

def i_from_k(k, N):
    """
    Given the kth element of N choose 2, return the row index i
    """
    total = -1 
    i = 0
    while total < k:
        total += N-i-1
        i += 1    
    return i-1, total

def j_from_total(k, total, N):
    return N-(total-k)-1 

def ij_from(k, N):
    i, total = i_from_k(k, N)
    j = j_from_total(k, total, N)
    return i, j

def k_from_ij(i, j, N):
    total = -1 
    for r in range(0, i):
        total += N - r-1
    k = total + (j-i) 
    return k 

def _test(N):
    k=0
    for i in range(N):
        for j in range(i+1, N):
            #print(i,j)
            a, b = ij_from(k, N)
            assert i == a, f"i{i}, a{a}"
            assert b == j, f"j{j}, b{b}"
            q = k_from_ij(i, j, N)
            print(i, j, a, b, q, k)
            assert q == k, (q, k)
            k+=1

def node_list2edge_list(nlist, N):
    nlist = sorted(nlist)
    combos = list(combinations(nlist, 2))
    return [k_from_ij(*edge, N=N) for edge in combos]

def weight2edge(x, slope=1000):
    """
    Map a weight in (0, 1) to a number peaked towards 0 and 1
    """
    return jax.nn.sigmoid((x-0.5)*slope)

def z2edge(z, slope=1000):
    return jax.nn.sigmoid((z)*slope)

class BaitPreyInfo(NamedTuple):
    edge_idxs : Array 
    Ndense : int 
    max_paths : int
    pc : float
   
class BaitPreyConnectivitySet(dist.Distribution):
    """
    Given a composite_pytree whose elements are
    - (edge_idxs, Ndense, maximal_paths, pc)  
    """
    def __init__(self, composite_pytree, dist_score_fn_dispatcher=simple_bait_prey_score_fn_dispatcher, bait_prey_slope=1.,
                 validate_args = None,):
        restraint_lst = []
        for composite in composite_pytree:
            edge_idxs, Ndense, mp, pc = composite
            restraint = BaitPreyConnectivity(
                    edge_idxs, Ndense, dist_score_fn_dispatcher=dist_score_fn_dispatcher,
                    maximal_shortest_path_to_calculate=mp,
                    pc=pc,
                    bait_prey_slope=bait_prey_slope)
            restraint_lst.append(restraint)
        self.restraint_lst = restraint_lst
        super(BaitPreyConnectivitySet, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
    def log_prob(self, edge_weights):
        value = 0
        for f in self.restraint_lst:
            value += f.log_prob(edge_weights)
        return value

class BaitPreyConnectivitySetWithRandomNormApproxN(dist.Distribution):
    """
    B approx  N(n * p, n * p * (p - 1))
    Given a composite_pytree whose elements are
    - (edge_idxs, Ndense, maximal_paths, pc)  
    """
    def __init__(self, composite_dict, Ntotal, 
                 validate_args = None,):
        restraint_lst = []
        cids = []
        n_dense = []
        probs = []
        for cid, composite in composite_dict.items():
            cids.append(cid)
            node_ids = composite['nodes']
            edge_idxs = jnp.array(node_list2edge_list(node_ids, Ntotal), dtype=jnp.int32)
            Ndense = composite['N']
            n_dense.append(Ndense)
            maximal_shortest_path_to_calculate = composite['maximal_shortest_path_to_calculate']
            p = composite['t']
            probs.append(p)
            restraint = BaitPreyConnectivityWithRandomN(
                    edge_idxs = edge_idxs,
                    Ndense = Ndense, 
                    maximal_shortest_path_to_calculate=maximal_shortest_path_to_calculate,
                    pc=p)
            restraint_lst.append(restraint)
        self.probs = probs
        self.cids = cids
        self.restraint_lst = restraint_lst
        self.n_restraints = len(restraint_lst)
        self.n_dense = n_dense
        super(BaitPreyConnectivitySetWithRandomNormApproxN, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
    def log_prob(self, aij):
        """
        for f in self.restraint_lst:
            value += f.log_prob(v)
        def body(i, val):
            carry, aij, Ns = val
            carry = carry + self.restraint_lst[i].log_prob(aij, Ns[i]) 
            return carry, aij, Ns
        aij, Ns = v
        """
        value = 0
        for i in range(self.n_restraints):
            cid = self.cids[i]
            n = self.n_dense[i]
            p = self.probs[i]
            N_i = numpyro.sample(f"N_{cid}",
                                 dist.Normal(n * p,
                                 n * p * (1 - p)))
            value += self.restraint_lst[i].log_prob((aij, N_i))
        return value #jax.lax.fori_loop(0, self.n_restraints, body, (0, aij, Ns)) 


def _example():
    # Don't nest the edge lists
    c1 = BaitPreyInfo(jnp.array([0, 1, 2]), 3, 8, 0.5)
    c2 = BaitPreyInfo(jnp.array([1, 2, 4]), 4, 8, 0.5) 
    composite_pytree = [c1, c2]
    return composite_pytree


class BaitPreyConnectivity(dist.Distribution):
    """
    The first index of the composite must be the bait
    Ensures that every prey is connected to the bait either directly or indirectly.
    Given an edge_weight_lst, an array of composite indices, and a bait_idx
    calculate D_composite up to N, the all pairs shortest paths distance matrix of composite connectivtiy.
    Score the average number of connected prey
    1. Given a list of nodes particle_idxs in the composite
    2. Get the edge weights at the list_values
    3. Create the dense matrix representation of the edge weights
    4. Map the matrix to binary protein interactions
    5. Calculate the all pairs shortest paths within the composite
    6. Get the list of shortests paths to the bait
    7. Count the number of paths with a distance less the d_max
    8. Score the number of connected prey
    M = N * (N-1) // 2 = 1/2 N **2 - 1/2 N
    2 M = N**2 - N
    Params:
      edge_idxs : an array of edge idxs
      Ndense    : the number of nodes in the composite (including bait)
      dist_score_fn : given a vector of distances to the bait, return a score
      maximal_shortest_paths_to_calculate :
        the maximal distance to calculate during Warshal Algorithm
      pc : prior probabiltiy of a composite
      binary_ppi_threshold : The descision boundrary for PPI such that
        a PPI is >= threshold.
    """
    def __init__(self,
        edge_idxs,
        Ndense,
        dist_score_fn_dispatcher=simple_bait_prey_score_fn_dispatcher,
        maximal_shortest_path_to_calculate=10,
        binary_ppi_threshold=0.5,
        pc = 1.0,
        bait_prey_slope=1.,
        validate_args=None):
        dist_score_fn = simple_bait_prey_score_fn_dispatcher(
                edge_idxs,
                Ndense,
                maximal_shortest_path_to_calculate,
                binary_ppi_threshold,
                pc,
                bait_prey_slope,)
        # Only one of edge ids or node ids should be set
        #if (node_idxs is None) ^ (edge_idxs is None):
        #    raise ValueError
        #N = m_from_n_choose2(M) 
        M = edge_idxs.shape[0]
        self.edge_idxs = edge_idxs 
        self.bait_idx_global = edge_idxs[0] 
        self.bait_idx_dense = 0 
        self.N = Ndense
        self.M = M
        self.maximal_shortest_path_to_calculate = maximal_shortest_path_to_calculate
        self.binary_ppi_threshold = binary_ppi_threshold
        self.dist_score_fn = dist_score_fn
        self.pc = pc
        super(BaitPreyConnectivity, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
        """
        Note : pc not implemented
        """
    
    def get_dense_matrix_from_edge_weight_lst(self, edge_weights):
        composite_edges = edge_weights[self.edge_idxs]
        adjacency_w = flat2matrix(composite_edges, self.N)
        return adjacency_w

    def weight2binary(self, a, DTYPE=jnp.int32):
        """
        Given an array a and a threshold, return the binary representation of a
        where all elements less than threshold are 0 and elements >= threshold are 1
        """
        threshold = self.binary_ppi_threshold
        return jnp.where(a < threshold, jnp.array([0], dtype=DTYPE), jnp.array([1], dtype=DTYPE))
    
    def apsp_up_to_dmax(self, A):
        return shortest_paths_up_to_N(A, self.maximal_shortest_path_to_calculate)

    def log_prob(self, edge_weights):
        A = self.get_dense_matrix_from_edge_weight_lst(edge_weights) # (N, N)
        A = self.weight2binary(A)  # (N, N)
        # Warshal O(N^3)
        D = self.apsp_up_to_dmax(A) # (N, N) 
        # New coordinate of the composite idx
        distance2bait = D[self.bait_idx_dense, :] #(N,)
        # Set the distance of a bait to itself as 0, bait must satisfy the restraint
        distance2bait = distance2bait.at[self.bait_idx_dense].set(0) #(N, )
        return self.dist_score_fn(distance2bait) * self.pc


class BaitPreyConnectivityWithRandomN(dist.Distribution):
    """
    The first index of the composite must be the bait
    Ensures that every prey is connected to the bait either directly or indirectly.
    Given an edge_weight_lst, an array of composite indices, and a bait_idx
    calculate D_composite up to N, the all pairs shortest paths distance matrix of composite connectivtiy.
    Score the average number of connected prey
    1. Given a list of nodes particle_idxs in the composite
    2. Get the edge weights at the list_values
    3. Create the dense matrix representation of the edge weights
    4. Map the matrix to binary protein interactions
    5. Calculate the all pairs shortest paths within the composite
    6. Get the list of shortests paths to the bait
    7. Count the number of paths with a distance less the d_max
    8. Score the number of connected prey
    M = N * (N-1) // 2 = 1/2 N **2 - 1/2 N
    2 M = N**2 - N
    Params:
      edge_idxs : an array of edge idxs
      Ndense    : the number of nodes in the composite (including bait)
      dist_score_fn : given a vector of distances to the bait, return a score
      maximal_shortest_paths_to_calculate :
        the maximal distance to calculate during Warshal Algorithm
      pc : prior probabiltiy of a composite
      binary_ppi_threshold : The descision boundrary for PPI such that
        a PPI is >= threshold.
    """
    def __init__(self,
        edge_idxs,
        Ndense,
        maximal_shortest_path_to_calculate=10,
        binary_ppi_threshold=0.5,
        pc = 1.0,
        bait_prey_slope=1.,
        validate_args=None):
        M = edge_idxs.shape[0]
        self.edge_idxs = edge_idxs 
        self.bait_idx_global = edge_idxs[0] 
        self.bait_idx_dense = 0 
        self.N = Ndense
        self.M = M
        self.maximal_shortest_path_to_calculate = maximal_shortest_path_to_calculate
        self.binary_ppi_threshold = binary_ppi_threshold
        self.pc = pc
        super(BaitPreyConnectivityWithRandomN, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
        """
        Note : pc not implemented
        """
    
    def get_dense_matrix_from_edge_weight_lst(self, edge_weights):
        composite_edges = edge_weights[self.edge_idxs]
        adjacency_w = flat2matrix(composite_edges, self.N)
        return adjacency_w

    def weight2binary(self, a, DTYPE=jnp.int32):
        """
        Given an array a and a threshold, return the binary representation of a
        where all elements less than threshold are 0 and elements >= threshold are 1
        """
        threshold = self.binary_ppi_threshold
        return jnp.where(a < threshold, jnp.array([0], dtype=DTYPE), jnp.array([1], dtype=DTYPE))
    
    def apsp_up_to_dmax(self, A):
        return shortest_paths_up_to_N(A, self.maximal_shortest_path_to_calculate)

    def log_prob(self, val):
        edge_weights, Nc = val
        A = self.get_dense_matrix_from_edge_weight_lst(edge_weights) # (N, N)
        A = self.weight2binary(A)  # (N, N)
        # Warshal O(N^3)
        D = self.apsp_up_to_dmax(A) # (N, N) 
        # New coordinate of the composite idx
        distance2bait = D[self.bait_idx_dense, :] #(N,)
        # Set the distance of a bait to itself as 0, bait must satisfy the restraint
        distance2bait = distance2bait.at[self.bait_idx_dense].set(0) #(N, )
        # Connectivity
        Sc = distance2bait < self.maximal_shortest_path_to_calculate
        # Score function
        x = jnp.sum(Sc) - Nc + 1  
        return jax.nn.sigmoid((x+0.5)*10.) # scaling the sigmoid function, how to set these weights?

class Histogram(dist.Distribution):
    def __init__(self, a, bins, density=True, validate_args=None):
        def histogram_piecewise_constant_pdf(x, bin_edges, bin_heights, n_bins):
            #bin_edges = jnp.array(bin_edges)
            #bin_heights = jnp.array(bin_heights)
            bin_indices = jnp.digitize(x, bin_edges) - 1
            bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
            return bin_heights[bin_indices]
        bin_heights, bin_edges = jnp.histogram(a, bins=bins, density=density)
        bin_heights = jnp.array(bin_heights)
        bin_edges = jnp.array(bin_edges)
        self.probs = bin_heights
        self.n_bins = len(bin_heights)
        self.bin_edges = bin_edges
        self.support = dist.constraints.real # Is this right?
        super(Histogram, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
        # Initialize the PDF
        pdf = jax.tree_util.Partial(histogram_piecewise_constant_pdf,
              bin_edges=self.bin_edges, bin_heights=self.probs,
              n_bins=self.n_bins)
        self.pdf = pdf
    def sample(self, key, sample_shape=()):
        bin_indices = jax.random.categorical(key, jnp.log(self.probs), shape=sample_shape)
        return self.bin_edges[bin_indices]
    def log_prob(self, value):
        return jnp.log(self.pdf(value))

class HistogramLowerBound(dist.Distribution):
    """
    As the Histogram class except all values below lower bound
    are clipped to min_val on the pdb scale such that score = log(min_val)
    """
    def __init__(self, a, bins, density=True, validate_args=None, lower_bound=0, min_val=0.001):
        def histogram_piecewise_constant_pdf(x, bin_edges, bin_heights, n_bins):
            #bin_edges = jnp.array(bin_edges)
            #bin_heights = jnp.array(bin_heights)
            bin_indices = jnp.digitize(x, bin_edges) - 1
            bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
            return bin_heights[bin_indices]
        bin_heights, bin_edges = jnp.histogram(a, bins=bins, density=density)
        bin_heights = jnp.array(bin_heights)
        bin_edges = jnp.array(bin_edges)
        self.probs = bin_heights
        self.n_bins = len(bin_heights)
        self.bin_edges = bin_edges
        self.support = dist.constraints.real # Is this right?
        super(Histogram, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
        # Initialize the PDF
        pdf = jax.tree_util.Partial(histogram_piecewise_constant_pdf,
              bin_edges=self.bin_edges, bin_heights=self.probs,
              n_bins=self.n_bins)
        self.pdf = pdf
        self._max_dens = np.max(probs)
    def sample(self, key, sample_shape=()):
        bin_indices = jax.random.categorical(key, jnp.log(self.probs), shape=sample_shape)
        return self.bin_edges[bin_indices]
    def log_prob(self, value):
        return jnp.log(jnp.clip(self.pdf(value), min_val, self._max_dens))

def matrix2flat(M, row_major=True):
    """
    An iterative method to flatten a matrix
    """
    n, m = M.shape
    N = math.comb(n, 2)
    #N = n * (n-1) // 2 
    a = jnp.zeros(N, dtype=M.dtype)
    k=0
    if row_major:
        for row in range(n):
            l = m - row - 1
            a = a.at[k:k + l].set(M[row, row + 1:m])
            k += l
    else:
        for col in range(m):
            l = n - col - 1 
            a = a.at[k:k + l].set(M[col+1:n, col])
            k += l 
    return a

def flat2matrix(a, n: int, row_major=True):
    """
    a : the input flattened array
    n : the number of columns in the matrix
    """
    M = jnp.zeros((n, n))
    n, m = M.shape
    k = 0
    if row_major:
        for row in range(n):
            l = m - row - 1
            M = M.at[row, row+1:m].set(a[k:k + l]) 
            k += l
    else:
        raise ValueError
    return M + M.T

def model14_data_getter(ntest=3005, from_pickle=True):
    if from_pickle:
        with open("model14_data.pkl", "rb") as f:
            d = pkl.load(f)
            return d
    # Load in AP-MS data and flatten
    with open("xr_apms_correlation_matrix.pkl", "rb") as f:
        apms_correlation_matrix = pkl.load(f)
    assert np.alltrue(apms_correlation_matrix == apms_correlation_matrix).T
    apms_correlation_matrix = apms_correlation_matrix[0:ntest, 0:ntest]
    n, m = apms_correlation_matrix.shape
    apms_tril_indices = jnp.tril_indices(n, k=-1)
    flattened_apms_similarity_scores = matrix2flat(
            jnp.array(apms_correlation_matrix.values, dtype=jnp.float32)) 
    assert len(flattened_apms_similarity_scores) == math.comb(n, 2)
    # Load in shuffled data and create null_hist_prob_dens
    with open("shuffled_apms_correlation_matrix.pkl", "rb") as f:
        shuffled_apms_correlation_matrix = pkl.load(f)
    shuffled_apms_correlation_matrix = shuffled_apms_correlation_matrix[0:ntest, 0:ntest]
    flattened_shuffled_apms = matrix2flat(
            jnp.array(shuffled_apms_correlation_matrix, dtype=jnp.float32))
    null_dist = Histrogram(flattened_shuffled_apms, bins=1000)
    return {"flattened_apms_similarity_scores" : flattened_apms_similarity_scores,
            "flattened_apms_shuffled_similarity_scores" : flattened_shuffled_apms}
            #"null_dist" : null_dist} 

def model22_ll_lp_data_getter(save_dir):
    with open(str(save_dir) + "data.pkl", "rb") as f:
        return pkl.load(f)

def model23_ll_lp_data_getter(save_dir):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    with open(str(save_dir / "modeler_vars.json"), 'r') as f:
        modeler_vars = json.load(f)
    dd = data_from_spec_table_and_composite_table(str(save_dir), modeler_vars['thresholds'])
    with open(save_dir / "shuffled_apms_correlation_matrix.pkl", "rb") as f:
        shuffled_apms_correlation_matrix = pkl.load(f)
    assert len(set(dd.keys()).intersection(modeler_vars.keys())) == 0
    dd = dd | modeler_vars
    dd['shuff_corr_all'] = shuffled_apms_correlation_matrix
    return dd

def model23_data_transformer(data_dict):
    """
    Transforms input data to be ready for modeling. Performs some checks.
    """
    dd = data_dict.copy()
    # Check inputs
    assert len(dd['corr'].index) == len(set(dd['corr'].index)) # Check all rows are uniquely indexed
    assert len(dd['corr'].index) > 2 # No point in modeling small networks
    assert dd['corr'].shape == dd['shuff_corr'].shape
    assert dd['corr'].ndim == 2
    r, c = dd['corr'].shape
    assert r == c
    dd = dd | {"N" : len(dd['corr'].index)}
    dd['M'] = dd['N'] * (dd['N'] - 1) // 2
    # map node names to integer identifiers 
    name2node_idx = {name : i for i, name in enumerate(dd['corr'].index)}
    node_idx2name = {i: name for name, i in name2node_idx.items()}
    dd['name2node_idx'] = name2node_idx
    dd['node_idx2name'] = node_idx2name
    #Remap names within composites to node_idxs 
    new_composite_dict_norm_approx = {}
    new_composite_dict_p_is_1 = {}
    for cid, c  in dd['composite_dict'].items():
        # Check the n, p criterion for valid binomial approximation
        n = len(c['nodes'])
        p = c['t']
        temp = [] 
        for name in c['nodes']:
            temp.append(name2node_idx[name])
        n_pairs = math.comb(len(temp), 2)
        temp_composite = {'nodes' : temp, 't' : p, 'SID' : c['SID'], 'N': len(temp), "n_pairs": n_pairs,'maximal_shortest_path_to_calculate': min(22, n_pairs)}
 
        if p == 1:
            new_composite_dict_p_is_1[cid] = temp_composite
        elif (p * n >= 5) and (n * (1-p) >= 5):
            new_composite_dict_norm_approx[cid] = temp_composite 
        else:
            logging.warning(f"Excluding Composite {cid}. Invalid Binomial Approximation. n: {n} p {p} np : {n * p}) n(1-p) : {n * (1 - p)}")
    dd['new_composite_dict_norm_approx'] = new_composite_dict_norm_approx 
    dd['new_composite_dict_p_is_1'] = new_composite_dict_p_is_1
    assert np.all(dd['corr'].index == dd['shuff_corr'].index) 
    # Flatten corr and shuff corr to jax arrays
    dd['apms_corr_flat'] = matrix2flat(jnp.array(dd['corr'].values, dtype=jnp.float32))
    #dd['apms_shuff_corr_flat'] = matrix2flat(jnp.array(dd['shuff_corr'].values, dtype=jnp.float32))
    dd['apms_shuff_corr_all_flat'] = matrix2flat(jnp.array(dd['shuff_corr_all'], dtype=jnp.float32))
    dd['n_composites'] = len(dd['composite_dict'])
    logging.info(f"N composites {dd['n_composites']}")
    dd.pop("composite_dict")

    #dd['N_per_composite'] = np.array([c['N'] for i, c in dd['composite_dict'].items()], dtype=jnp.int32)
    #dd['p_per_composite'] = np.array([c['t'] for i, c in dd['composite_dict'].items()], dtype=jnp.float32)
    assert len(dd['apms_corr_flat']) == dd['M']
    return dd 

def preyu2uid_mapping_dict():
    apms_data_path = "../table1.csv"
    apms_ids = pd.read_csv(apms_data_path)
    # 1 to 1 mapping
    reference_preyu2uid = {r['PreyGene'].removesuffix('_HUMAN'): r['UniprotId'] for i,r in apms_ids.iterrows()}
    return reference_preyu2uid

def model14(model_data):
    """
    M : pT_{i, j} edge weights
    """
    #assert data.shape[0] == N, "The number of data points must be equal to N"
    # Load in variables from model data to allow proper tracing
    flattened_apms_similarity_scores = model_data["flattened_apms_similarity_scores"]
    flattened_apms_similarity_scores = jnp.array(
            flattened_apms_similarity_scores, dtype=jnp.float32)
    # Test line
    #ntest = model_data['ntest']
    flattened_apms_similarity_scores = flattened_apms_similarity_scores#[0:ntest]
    shuffled = model_data["flattened_apms_shuffled_similarity_scores"]#[0:ntest]
    N = flattened_apms_similarity_scores.shape[0]
    #null_dist = model_data['null_dist']
    #null_dist = null_dist.expand([N,])
    #mixture_probs = numpyro.sample("mixture_probs", dist.Dirichlet(jnp.ones(K)).expand([N,]))
    # Prior edge probability is 0.67
    pT = numpyro.sample("pT", dist.Uniform(0, 1).expand([N,]))
    #pT = numpyro.sample("pT", dist.Beta(4, 2).expand([N,])) # Mean is 0.66
    mixture_probs = jnp.array([pT, 1-pT]).T
    # 1. Null distribution
    null_dist = Histogram(shuffled, bins=1000).expand([N,])
    # Hyper priors 0.23 and 0.22 are set from previous modeling
    causal_dist = dist.Normal(0.23, 0.22).expand([N,]) 
    #means = numpyro.sample("means", dist.Normal(0, 5).expand([N, K]))
    #stds = numpyro.sample("stds", dist.HalfNormal(5).expand([N, K]))
    #components = dist.Normal(means, stds)
    mixtures = dist.MixtureGeneral(dist.Categorical(probs=mixture_probs),
                                  [causal_dist, null_dist])
    numpyro.sample("obs", mixtures, obs=flattened_apms_similarity_scores)

def model15(model_data):
    """"
    Model 15 is exactly the same as model 14 except we use a non uniform prior
    for the edge weight
    """
    """
    M : pT_{i, j} edge weights
    """
    #assert data.shape[0] == N, "The number of data points must be equal to N"
    # Load in variables from model data to allow proper tracing
    flattened_apms_similarity_scores = model_data["flattened_apms_similarity_scores"]
    flattened_apms_similarity_scores = jnp.array(
            flattened_apms_similarity_scores, dtype=jnp.float32)
    # Test line
    #ntest = model_data['ntest']
    flattened_apms_similarity_scores = flattened_apms_similarity_scores#[0:ntest]
    shuffled = model_data["flattened_apms_shuffled_similarity_scores"]#[0:ntest]
    N = flattened_apms_similarity_scores.shape[0]
    #null_dist = model_data['null_dist']
    #null_dist = null_dist.expand([N,])
    #mixture_probs = numpyro.sample("mixture_probs", dist.Dirichlet(jnp.ones(K)).expand([N,]))
    # Prior edge probability is 0.67
    # Hyperprior set by plotting Beta Distribution during Synthetic Benchmark50 notebook
    pT = numpyro.sample("pT", dist.Beta(0.5, 0.5)) 
    #pT = numpyro.sample("pT", dist.Beta(4, 2).expand([N,])) # Mean is 0.66
    mixture_probs = jnp.array([pT, 1-pT]).T
    # 1. Null distribution
    null_dist = Histogram(shuffled, bins=1000).expand([N,])
    # Hyper priors 0.23 and 0.22 are set from previous modeling
    causal_dist = dist.Normal(0.23, 0.22).expand([N,]) 
    #means = numpyro.sample("means", dist.Normal(0, 5).expand([N, K]))
    #stds = numpyro.sample("stds", dist.HalfNormal(5).expand([N, K]))
    #components = dist.Normal(means, stds)
    mixtures = dist.MixtureGeneral(dist.Categorical(probs=mixture_probs),
                                  [causal_dist, null_dist])
    numpyro.sample("obs", mixtures, obs=flattened_apms_similarity_scores)

def model16(model_data):
    """"
    Model 15 is exactly the same as model 14 except we use a non uniform prior
    for the edge weight

    Use lower bounds for edge weights
    M : pT_{i, j} edge weights
    """
    #assert data.shape[0] == N, "The number of data points must be equal to N"
    # Load in variables from model data to allow proper tracing
    flattened_apms_similarity_scores = model_data["flattened_apms_similarity_scores"]
    flattened_apms_similarity_scores = jnp.array(
            flattened_apms_similarity_scores, dtype=jnp.float32)
    # Test line
    #ntest = model_data['ntest']
    flattened_apms_similarity_scores = flattened_apms_similarity_scores#[0:ntest]
    shuffled = model_data["flattened_apms_shuffled_similarity_scores"]#[0:ntest]
    N = flattened_apms_similarity_scores.shape[0]
    #null_dist = model_data['null_dist']
    #null_dist = null_dist.expand([N,])
    #mixture_probs = numpyro.sample("mixture_probs", dist.Dirichlet(jnp.ones(K)).expand([N,]))
    # Prior edge probability is 0.67
    # Hyperprior set by plotting Beta Distribution during Synthetic Benchmark50 notebook
    pT = numpyro.sample("pT", dist.Beta(0.5, 0.5)) 
    #pT = numpyro.sample("pT", dist.Beta(4, 2).expand([N,])) # Mean is 0.66
    mixture_probs = jnp.array([pT, 1-pT]).T
    # 1. Null distribution
    null_dist = Histogram(shuffled, bins=1000).expand([N,])
    # Hyper priors 0.23 and 0.22 are set from previous modeling
    causal_dist = dist.Normal(0.23, 0.22).expand([N,]) 
    #means = numpyro.sample("means", dist.Normal(0, 5).expand([N, K]))
    #stds = numpyro.sample("stds", dist.HalfNormal(5).expand([N, K]))
    #components = dist.Normal(means, stds)
    mixtures = dist.MixtureGeneral(dist.Categorical(probs=mixture_probs),
                                  [causal_dist, null_dist])
    numpyro.sample("obs", mixtures, obs=flattened_apms_similarity_scores)

def model17_test():
    a = jnp.array([0, 1, 5, 8, 7, 9])
    e = numpyro.sample("pT", dist.Beta(0.5, 0.5).expand([50,])) 
    cc = CompositeConnectivity(a, 4)
    cc1_ld = cc.log_prob(e)
    numpyro.factor('cc1_ld', cc1_ld)

def model18_test(w):
    a = jnp.array([0, 1, 5, 8, 7, 9])
    e = numpyro.sample("pT", dist.Beta(w, w).expand([50,])) 
    cc = CompositeConnectivity(a, 4)
    cc1_ld = cc.log_prob(e)
    numpyro.factor('cc1_ld', cc1_ld)

def model18_test2(w):
    a = jnp.array([0, 1, 5, 8, 7, 9])
    e = numpyro.sample("pT", dist.Beta(w, w).expand([50,])) 
    e = jnp.clip(e, a_min=0.001, a_max=0.999)
    cc = CompositeConnectivity(a, 4)
    cc1_ld = cc.log_prob(e)
    numpyro.factor('cc1_ld', cc1_ld)

def model18_test3(w, scale=1):
    a = jnp.arange(28) 
    e = numpyro.sample("pT", dist.Beta(w, w).expand([50,])) 
    e = jnp.clip(e, a_min=0.001, a_max=0.999)
    cc = CompositeConnectivity(a, 8, scale=scale)
    cc1_ld = cc.log_prob(e)
    numpyro.factor('cc1_ld', cc1_ld)

def model18_test4(w, scale=1):
    x = numpyro.sample('x', dist.Uniform(-10, 10))
    x_lp = dist.Normal().log_prob(x)
    numpyro.factor('x_lp', x_lp)


def model18_test5():
    probs = jnp.ones(50) * 0.5
    d = dist.Bernoulli(probs=probs)
    with numpyro.plate('edges', 50):
        e = numpyro.sample('e', d, infer={'enumerate': 'parallel'})

def model18_test6(w, scale, mag):
    a = jnp.arange(153) 
    e = numpyro.sample("pT", dist.Beta(w, w).expand([500,])) 
    e_at_composite = e[a]
    x = jnp.sum(e_at_composite)
    numpyro.deterministic('x', x)
    e = jnp.clip(e, a_min=0.001, a_max=0.999)
    cc = CompositeConnectivity(a, 18, scale=scale, magnitude=mag)
    cc1_ld = cc.log_prob(e)
    numpyro.factor('cc1_ld', cc1_ld)

def model18_test7():
    """
    Keep the basal frequency of edges low
    """
    # Imagine a single composite
    z = numpyro.sample("pT", dist.Beta(0.01, 0.4).expand([6,]))
    e = jax.nn.sigmoid((z-0.5)*20) 
    numpyro.deterministic('e', e)
    x = jnp.sum(e) # sum of edge weights
    # Composite connectivity
    y = jax.nn.sigmoid((x-4+1)*jnp.square(4))
    numpyro.factor('y', y)

def model19_test1(m_total, n_cc1, m_cc1, sig_scale=100, cc_scale=4):
    # Latent unconstrained edge weights
    z = numpyro.sample('z', dist.Normal().expand([m_total,]))
    e = jax.nn.sigmoid(z * sig_scale)
    a = jnp.arange(m_cc1)
    cc = CompositeConnectivity(a, n_cc1, scale=cc_scale)
    e_at_cc1 = e[a]
    numpyro.deterministic('e_at_cc1', e_at_cc1)
    numpyro.deterministic('x', jnp.sum(e_at_cc1))
    numpyro.factor('cc1_lp', cc.log_prob(e))

def model19_test2(m_total, n_cc1, m_cc1, sig_scale=100, cc_scale=4):
    # Latent unconstrained edge weights
    z = numpyro.sample('z', dist.Normal().expand([m_total,]))
    e = jax.nn.sigmoid(z * sig_scale)
    a = jnp.arange(m_cc1)
    b = jnp.array([0, 1, 18, 19, 8, 17]) 
    c = jnp.array([8, 15, 1]) 
    d = jnp.array([0])
    cc = CompositeConnectivity(a, n_cc1, scale=cc_scale)
    cc2 = CompositeConnectivity(b, 4, scale=cc_scale) 
    cc3 = CompositeConnectivity(c, 3, scale=cc_scale)
    cc4 = CompositeConnectivity(d, 2, scale=cc_scale)
    e_at_cc1 = e[a]
    e_at_cc2 = e[b]
    numpyro.deterministic('e_at_cc1', e_at_cc1)
    numpyro.deterministic('e_at_cc2', e_at_cc2)
    numpyro.deterministic('x', jnp.sum(e_at_cc1))
    numpyro.deterministic('y', jnp.sum(e_at_cc2))
    numpyro.factor('cc1_lp', cc.log_prob(e))
    numpyro.factor('cc2_lp', cc2.log_prob(e))
    numpyro.factor('cc3_lp', cc3.log_prob(e))
    numpyro.factor('cc4_lp', cc4.log_prob(e))

def model20_test1():
    # (5, 5) graph
    N = 5
    M =  N * (N-1) // 2
    z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1
    c1 = jnp.array([0, 1, 2])
    c2 = jnp.array([1, 2, 3])
    bp1 = BaitPreyConnectivity(c1)
    bp2 = BaitPreyConnectivity(c2)
    s1 = bp1.log_prob(e)
    s2 = bp2.log_prob(e)
    numpyro.factor('bp1', s1) 
    numpyro.factor('bp2', s2) 

def model20_test2():
    N = 20
    # (20, 20) graph
    M = N * (N-1) // 2
    z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1
    c1 = jnp.array([0, 1, 2])
    c2 = jnp.array([1, 2, 3, 4])
    c3 = jnp.array([1, 2, 7, 9, 11]) 
    bp1 = BaitPreyConnectivity(c1)
    bp2 = BaitPreyConnectivity(c2)
    bp3 = BaitPreyConnectivity(c3)

    s1 = bp1.log_prob(e)
    s2 = bp2.log_prob(e)
    s3 = bp3.log_prob(e)
    numpyro.factor('bp1', s1) 
    numpyro.factor('bp2', s2) 
    numpyro.factor('bp3', s3)

def model20_test3():
    N = 20
    # (20, 20) graph
    M = N * (N-1) // 2
    z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1
    c1 = jnp.array([0, 1, 2])
    c2 = jnp.array([1, 2, 3, 4])
    c3 = jnp.array([1, 2, 7, 9, 11]) 
    c4 = jnp.array([1, 2])
    bp1 = BaitPreyConnectivity(c1)
    bp2 = BaitPreyConnectivity(c2)
    bp3 = BaitPreyConnectivity(c3)
    bp4 = BaitPreyConnectivity(c4)
    s1 = bp1.log_prob(e)
    s2 = bp2.log_prob(e)
    s3 = bp3.log_prob(e)
    s4 = bp4.log_prob(e)
    numpyro.factor('bp1', s1) 
    numpyro.factor('bp2', s2) 
    numpyro.factor('bp3', s3)
    numpyro.factor('bp4', s4)

def model20_test4():
    N = 3 
    # (20, 20) graph
    M = N * (N-1) // 2
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1
    c4 = [0, 1]
    e4 = jnp.array(node_list2edge_list(c4, N))
    bp4 = BaitPreyConnectivity(e4, 2)
    s4 = bp4.log_prob(e)
    numpyro.factor('bp4', s4)

def model20_test5():
    score_fn = Partial(simple_bait_prey_score_fn, slope=2.)
    N = 3 
    # (20, 20) graph
    M = N * (N-1) // 2
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1
    c4 = jnp.array([0, 1])
    bp4 = BaitPreyConnectivity(c4, dist_score_fn=score_fn)
    s4 = bp4.log_prob(e)
    numpyro.factor('bp4', s4)

def model20_test6a(eidx, slope):
    score_fn = Partial(simple_bait_prey_score_fn, slope=slope)
    N = 3 
    # (20, 20) graph
    M = N * (N-1) // 2
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    ec4 = jnp.array([eidx])
    # composite 2 [1, 2]
    #ec1 = jnp.array([2])
    bp4 = BaitPreyConnectivity(ec4, Ndense=2, dist_score_fn=score_fn)
    #bp1 = BaitPreyConnectivity(ec1, dist_score_fn=score_fn) 
    s4 = bp4.log_prob(e)
    #s1 = bp1.log_prob(e)
    numpyro.factor('bp4', s4)
    #numpyro.factor('bp1', s1)

def model20_test6b(eidx, slope):
    # Globals
    N = 3 
    score_fn = Partial(simple_bait_prey_score_fn, slope=slope)
    M = N * (N-1) // 2
    # Composite 0
    ec0 = np.array(node_list2edge_list([0, 1], N))
    bp0 = BaitPreyConnectivity(ec0, Ndense=2, dist_score_fn=score_fn)
    # Composite 1
    ec1 = np.array(node_list2edge_list([0, 2], N)) 
    bp1 = BaitPreyConnectivity(ec1, Ndense=2, dist_score_fn=score_fn)
    # Scoring
    # Prior edge frequency
    z = numpyro.sample('z', dist.Normal(loc=0.4).expand([M,]))
    e = weight2edge(z)
    numpyro.deterministic('e', e)
    # Prior composite connectivity  
    s0 = bp0.log_prob(e)
    s1 = bp1.log_prob(e)
    numpyro.factor('bp0', s0)
    numpyro.factor('bp1', s1)

def model20_test6c(slope):
    # V={0,1,2, 3} Cm={0, 1, 2, 3}
    cm = [0, 1, 2, 3]
    score_fn = Partial(simple_bait_prey_score_fn, slope=slope)
    N = 4 
    # (20, 20) graph
    M = N * (N-1) // 2
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    #e = jax.nn.sigmoid((z-0.5) * 100)
    e = weight2edge(z)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    ec4 = np.array(node_list2edge_list(cm, N)) 
    # composite 2 [1, 2]
    bp4 = BaitPreyConnectivity(ec4, Ndense=4, dist_score_fn=score_fn)
    s4 = bp4.log_prob(e)
    numpyro.factor('bp4', s4)

def model20_test6d(slope):
    # V={0,1,2, 3} C0={0, 1, 2} C1={1,2,3}
    N = 4 
    M = N * (N-1) // 2
    c0 = [0, 1, 2]
    c1 = [1, 2, 3]
    e0 = np.array(node_list2edge_list(c0, N))
    e1 = np.array(node_list2edge_list(c1, N))
    score_fn = Partial(simple_bait_prey_score_fn, slope=slope) 
    # (20, 20) graph
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = weight2edge(z)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    # composite 2 [1, 2]
    bp0 = BaitPreyConnectivity(e0, Ndense=3, dist_score_fn=score_fn)
    bp1 = BaitPreyConnectivity(e1, Ndense=3, dist_score_fn=score_fn) 
    s0 = bp0.log_prob(e)
    s1 = bp1.log_prob(e)
    numpyro.factor("s0", s0)
    numpyro.factor("s1", s1)

def model20_test6f(N, Cm, slope=8.,):
    # V={0,1,2, 3} C0={0, 1, 2} C1={1,2,3}
    Ndense=len(Cm)
    M = N * (N-1) // 2
    e0 = np.array(node_list2edge_list(Cm, N))
    score_fn = Partial(simple_bait_prey_score_fn, slope=slope) 
    # (20, 20) graph
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = weight2edge(z)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    # composite 2 [1, 2]
    bp0 = BaitPreyConnectivity(e0, Ndense=Ndense, dist_score_fn=score_fn)
    s0 = bp0.log_prob(e)
    numpyro.factor("s0", s0)

def model20_test6f_null(N, Cm, slope=8.,):
    # V={0,1,2, 3} C0={0, 1, 2} C1={1,2,3}
    Ndense=len(Cm)
    M = N * (N-1) // 2
    e0 = np.array(node_list2edge_list(Cm, N))
    score_fn = Partial(simple_bait_prey_score_fn, slope=slope) 
    # (20, 20) graph
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = weight2edge(z)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    # composite 2 [1, 2]
    bp0 = BaitPreyConnectivity(e0, Ndense=Ndense, dist_score_fn=score_fn)


def model20_test6e_1a(slope, yscale):
    """
    Prior with equal edge frequency
    """
    # V={0,1,2, 3}
    # Cm={0, 1, 2}
    # 0 : (0, 1)
    # 1 : (0, 2)
    # 2 : (0, 3)
    # 3 : (1, 2)
    # 4 : (1, 3)
    # 5 : (2, 3)
    N = 4 
    score_fn = Partial(sigmoid_bait_prey_score_fn, slope=slope, N=N, yscale=yscale)
    # (20, 20) graph
    M = N * (N-1) // 2
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    # cm = {0, 1, 2}
    ec1 = jnp.array([0, 1,3]) 
    # cm = {1, 2, 3}
    bp1 = BaitPreyConnectivity(ec1, Ndense=3, dist_score_fn=score_fn)
    s1 = bp1.log_prob(e)
    numpyro.factor('bp1', s1)


def model20_test6e_1b():
    # V={0,1,2, 3}
    # Cm={0, 1, 2}
    # 0 : (0, 1)
    # 1 : (0, 2)
    # 2 : (0, 3)
    # 3 : (1, 2)
    # 4 : (1, 3)
    # 5 : (2, 3)
    N = 4 
    Ndense=3
    yscale=20.
    slope=10.
    score_fn = Partial(sigmoid_bait_prey_score_fn, slope=slope, N=Ndense, yscale=yscale)
    # (20, 20) graph
    M = N * (N-1) // 2
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    # cm = {0, 1, 2}
    ec1 = jnp.array([0, 1,3]) 
    # cm = {1, 2, 3}
    bp1 = BaitPreyConnectivity(ec1, Ndense=Ndense, dist_score_fn=score_fn)
    s1 = bp1.log_prob(e)
    numpyro.factor('bp1', s1)

def data_from_spec_table_and_composite_table(data_path, ms_thresholds, sep="\t", rseed=0):
    if isinstance(data_path, str):
        spec_path = Path(data_path) / "spec_table.tsv" 
        composite_path = Path(data_path) / "composite_table.tsv"
    composite_table = pd.read_csv(composite_path, sep=sep)
    # Remove prey below threshold
    min_threshold = np.min(ms_thresholds)
    assert min_threshold >= 0
    assert np.all(np.isreal(composite_table['MSscore'].values))
    assert np.all(composite_table['MSscore'] >= 0)
    sel = composite_table['MSscore'] >= min_threshold
    composite_table = composite_table[sel]
    # Get the composite, threshold pairs 
    sid_set = set(composite_table['SID'])
    composites = {}
    k=0
    for threshold in ms_thresholds:
        for sid in sid_set:
            sel = (composite_table['SID'] == sid) & (composite_table['MSscore'] >= threshold)
            if np.sum(sel) > 0:
                d = composite_table[sel]
                bait_l = list(set(d['Bait']))
                msscore_l = list(set(d['MSscore']))
                assert len(bait_l) == 1, bait_l
                #Ensure bait is first index
                prey_l = list(set(d['Prey'])) 
                if bait_l[0] in prey_l:
                    prey_l.remove(bait_l[0])
                nodes = bait_l + prey_l
                msscore = msscore_l[0]
                composites[k] = {'nodes':nodes, 't': threshold, 'SID': sid}
                k+=1
    spec_table = pd.read_csv(spec_path, sep=sep, index_col=0, header=None)
    # Remove the IDS that don't pass the minimal MSscore threshold
    prey_set = set(composite_table['Prey'])
    bait_set = set(composite_table['Bait'])
    all_ids = list(prey_set.union(bait_set))
    sel = []
    for name in spec_table.index: 
        val = True if name in all_ids else False
        sel.append(val)
    spec_table = spec_table[sel]
    #Calculate shuffled profile similiarties from all the data
    null_sc = spec_table.copy()
    key = jax.random.PRNGKey(rseed)
    shuff = jax.random.permutation(key, null_sc.values, axis=1) #shuffle rows
    null_sc.loc[:, :] = np.array(shuff) 
    shuff_corr = np.corrcoef(null_sc.values, rowvar=True)
    corr = np.corrcoef(spec_table.values, rowvar=True)
    shuff_corr = pd.DataFrame(shuff_corr, columns=spec_table.index, index=spec_table.index)
    corr = pd.DataFrame(corr, columns=spec_table.index, index=spec_table.index)
    return {"corr" : corr, "shuff_corr" : shuff_corr, "composite_dict" : composites} 

def row_mag(A):
    return np.sqrt(np.sum(np.square(A), axis=1))

def model21_test():
    score_fn = Partial(simple_bait_prey_score_fn, slope=slope)
    N = 3 
    # (20, 20) graph
    M = N * (N-1) // 2
    #z = numpyro.sample('z', dist.Beta(0.01, 0.04).expand([M,]))
    z = numpyro.sample('z', dist.Normal(loc=0.5).expand([M,]))
    e = jax.nn.sigmoid((z-0.5) * 100)
    numpyro.deterministic('e', e)
    # composite1 [0, 1]
    ec4 = jnp.array([eidx])
    # composite 2 [1, 2]
    #ec1 = jnp.array([2])
    bp4 = BaitPreyConnectivity(ec4, Ndense=2, dist_score_fn=score_fn)
    #bp1 = BaitPreyConnectivity(ec1, dist_score_fn=score_fn) 
    s4 = bp4.log_prob(e)
    #s1 = bp1.log_prob(e)
    numpyro.factor('bp4', s4)
    #numpyro.factor('bp1', s1)

def model22_lp(data):
    ...

def model22_ll(data):
    ...

def model22_ll_lp(data):
    """
    """
    # Unpack input information
    N = data["N"]
    z2edge_slope = data['z2edge_slope']
    composites = data['composites'] 
    lower_edge_prob = data['lower_edge_prob_bound']
    upper_edge_prob = data['upper_edge_prob_bound']
    bait_prey_slope = data['BAIT_PREY_SLOPE']
    flattened_apms_similarity_scores = data["flattened_apms_similarity_scores"]
    flattened_apms_similarity_scores = jnp.array(
            flattened_apms_similarity_scores, dtype=jnp.float32)
    shuffled = data["flattened_apms_shuffled_similarity_scores"]
    # Define Model Globals 
    #score_fn = Partial(simple_bait_prey_score_fn, slope=bait_prey_slope)
    M = N * (N-1)//2
    bait_prey_connectivity_set = BaitPreyConnectivitySet(composites, bait_prey_slope=bait_prey_slope)
    causal_dist = dist.Normal(0.23, 0.22).expand([M,]) 
    null_dist = Histogram(shuffled, bins=1000).expand([M,]) # ToDo Improve Null Dist
    # Scoring 
    pN = numpyro.sample('pN', dist.Uniform(low=lower_edge_prob, high=upper_edge_prob)) # (1,) # Base Rate
    mu = edge_prob2mu(pN)  
    numpyro.deterministic('mu', mu)
    z = numpyro.sample('z', dist.Normal(mu+0.5).expand([M,])) #(M, ) shifted normal, sigma must be 1
    # Edges
    aij = jax.nn.sigmoid((z-0.5)*z2edge_slope)
    # Prior
    bps_lp = bait_prey_connectivity_set.log_prob(aij)
    numpyro.factor("bps_lp", bps_lp)
    numpyro.deterministic("lp_score", bps_lp)
    mixture_probs = jnp.array([aij, 1-aij]).T
    # Profile similarity likelihood
    mixtures = dist.MixtureGeneral(
            dist.Categorical(probs=mixture_probs),
            [causal_dist, null_dist])
    #numpyro.sample("obs", mixtures, obs=flattened_apms_similarity_scores)

def silly_model(data):
    b = data['b']
    for i in range(10):
        #N_i = numpyro.sample(
        #      f'Ns_{i}',
        #      dist.BinomialProbs(probs = 0.7, total_count = i),
        #              infer = {"enumerate" : "parallel"})
        x = numpyro.sample(f'a_{i}', dist.Normal(3. - i))
    

def silly_model2(data):
    numpyro.sample('a', dist.Normal(data['a']))

def silly_model3(data):
    b = data['b']
    a = data['a']
    mu = jnp.zeros(10)
    a = jnp.array([1, 10, 100])
    b = jnp.array([0.2, 0.5, 0.8])
    for i in range(len(b)):
        with numpyro.plate(f'data_{i}', 1):
            k_i = numpyro.sample(f'k_{i}', dist.Binomial(a[i], b[i]), infer = {"enumerate" : "parallel"})

def mini_bait_prey_connectivity_with_random_N_norm_approx(data):
    composite_dict = {0 : {"nodes" : [0, 1, 4, 6, 9], "t": 0.5, "N" : 5, "maximal_shortest_path_to_calculate" : 15}}
    Ntotal = 11
    M = Ntotal * (Ntotal-1) // 2
    bait_prey_connectivity_set = BaitPreyConnectivitySetWithRandomNormApproxN(composite_dict, Ntotal)
    pN = numpyro.sample('pN', dist.Uniform(low=0, high=0.5))
    mu = edge_prob2mu(pN) # determinsitc transform
    numpyro.deterministic('mu', mu)
    z = numpyro.sample('z', dist.Normal(mu+0.5).expand([M,])) #(M, ) shifted normal, sigma must be 1. latent edges
    aij = jax.nn.sigmoid((z-0.5)*10)
    score = bait_prey_connectivity_set.log_prob(aij)
    numpyro.factor("score", score)

def mini_model(d):
    N = d['N']
    M = N * (N-1) // 2
    R0 = d['apms_shuff_corr_flat']
    R = d['apms_shuff_flat']
    null_dist = Histogram(R0, bins=500).expand([M,]) # Normalized
    causal_dist = dist.Normal(0.23, 0.22).expand([M,]) 
    aij = numpyro.sample('aij', dist.Normal(0, 1).expand([M]))
    # Null Hypothesis Test
    #ll_0 = null_dist.log_prob(aij)
    #ll_1 = causal_dist.log_prob(aij)
    mixture_probs = jnp.array([aij, 1-aij]).T
    # Profile similarity likelihood
    mixtures = dist.MixtureGeneral(
            dist.Categorical(probs=mixture_probs),
            [causal_dist, null_dist])
    numpyro.sample("obs", mixtures, obs=R)


def mini_nuts_run(rseed, model, data):
    nuts = NUTS(model)
    mcmc = MCMC(nuts, num_warmup=500, num_samples=500)
    key = jax.random.PRNGKey(rseed)
    mcmc.run(key, data)
    return mcmc


def model_23_ll_lp_init_to_zero_strategy(model_data):
    N = model_data['N']
    M = N * (N - 1) // 2
    return init_to_value(values = {"z" : jnp.ones(M) * 0.5, # latent space centered at 0.5
                                            "pN": 0.5})

def model_23_ll_lp_init_to_data_strategy(model_data):
    N = model_data['N']
    M = N * (N - 1) // 2
    R = model_data['apms_corr_flat']
    R = R + 0.5
    return init_to_value(values = {"z" : R, # latent space centered at 0.5
                                           })


def model23_ll_lp(model_data):
    # Unpack data for tracing
    d = model_data
    N = d['N'] 
    M = N * (N-1) // 2
    alpha = d['lower_edge_prob']
    beta = d['upper_edge_prob']
    composite_dict_p_is_1 = d['new_composite_dict_p_is_1']  # Can use N directl
    composite_dict_norm_approx = d['new_composite_dict_norm_approx'] # Can use the Binomial approximation 
    R = d['apms_corr_flat']
    # Note - here we clip the data for negative correlations
    # How to handle statistically significant negative correlations? What do they mean?
    # Likey don't enrich for PPI's
    # What does a statistically significant negative correlation mean? 
    R = np.clip(R, 0, 1.0)

    n_composites = d['n_composites']
    #N_per_composite = d['N_per_composite'] # (n_composites,)
    #p_per_composite = d['p_per_composite'] # (n_composites,)
    z2edge_slope = 1_000
    # Representation and Scoring
    u = numpyro.sample('pN', dist.Uniform(low=alpha, high=beta))
    mu = edge_prob2mu(u) # determinsitc transform
    numpyro.deterministic('mu', mu)
    z = numpyro.sample('z', dist.Normal(mu+0.5).expand([M,])) #(M, ) shifted normal, sigma must be 1. latent edges
    aij = jax.nn.sigmoid((z-0.5)*z2edge_slope)
    # Define things per composite
    #bait_prey_score("test", aij, c22_nodes, c22_N, N, 8, c22_t) 
    for k in range(0, 20):
        kth_bait_prey_score_norm_approx(aij, k, composite_dict_norm_approx, N)
    for k in range(21, 26):
        kth_bait_prey_score_norm_approx(aij, k, composite_dict_norm_approx, N)
    for k in range(27, 2):
        kth_bait_prey_score_norm_approx(aij, k, composite_dict_norm_approx, N)

    kth_bait_prey_score_norm_approx(aij, 31, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 32, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 34, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 35, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 37, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 38, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 42, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 45, composite_dict_norm_approx, N)
    kth_bait_prey_score_norm_approx(aij, 48, composite_dict_norm_approx, N)

    kth_bait_prey_score_p_is_1(aij, 50, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 51, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 52, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 53, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 54, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 55, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 56, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 57, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 58, composite_dict_p_is_1, N)
    kth_bait_prey_score_p_is_1(aij, 59, composite_dict_p_is_1, N)

    # Data Likelihood
    #R0 = d['apms_shuff_corr_all_flat']  # Use all the AP-MS data
    #null_dist = Histogram(R0, bins=100).expand([M,]) # Normalized
    #null_log_like = null_dist.log_prob(R)
    #INFINITY_FACTOR = 10 
    #causal_dist = dist.Normal(0.23, 0.22).expand([M,]) 
    # Null Hypothesis Test
    #ll_0 = null_dist.log_prob(aij)
    #ll_1 = causal_dist.log_prob(aij)

    # Score approximates log[(1-a) * p(R | H0)]
    #score = (1-aij)*null_log_like # Addition on the log scale
    #numpyro.factor("R", score)

    #mixture_probs = jnp.array([aij, 1-aij]).T
    # Profile similarity likelihood
    #mixture_model = dist.MixtureGeneral(
    #        dist.Categorical(probs=mixture_probs),
    #        [causal_dist, null_dist])
    #R_ll = mixture_model.log_prob(R)
    #numpyro.factor("R_ll", R_ll)
    ##numpyro.sample("obs", mixtures, obs=R)

def _note():
    """
    We need to create a restraint for Composites.

    1. Singlet function f(A)
    """

def call_and_prod(z):
    val = 1
    for i in z:
        val *= i()
    return val

def call_and_sum(x, z):
    val = 1
    for i in z:
        val += i(x)
    return val

def call_and_sum2(x, z, len_z):
    def body(i, v):
        x, val, z = v
        f = z[i]
        val += f(x)
        v = x, val, z
        return v
    return jax.lax.fori_loop(0, len_z, body, (x, 0, z))

def edge_prob2mu(edge_prob):
    """
    Given an independant edge probababilty, return
    the mean of a normal distribution with standard deviation 1.
    Such that the area from 0.5 to inf equals the edge probability.
    $F_x$ is the CDF
    $$ a = 1 - F_x(0) $$
    $$ 2 \times (1 - a)-1 = erf(\frac_{-\mu}{\sigma \sqrt{2}}) $$
    $$ t = erf^{-1} (2 \times (1-a) -1) $$
    $$ mu = - \sqrt(2) t $$
    """
    s = 2 * (1-edge_prob) - 1
    t = jsp.special.erfinv(s) 
    mu = -jnp.sqrt(2) * t
    return mu

def load(fpath):
    with open(fpath, 'rb') as f:
        dat = pkl.load(f)
    return dat

#Keys are model name, values are model_func, init_func pairs
_model_dispatch = {
    "model_10_wt": (model_10_wt, get_cov_model9_init_strategy, lambda x: int(x)),
    "model_10_vif": (model_10_vif, get_cov_model9_init_strategy, lambda x: int(x)),
    "model_10_mock": (model_10_mock, get_cov_model9_init_strategy, lambda x: int(x)),
    "model_10_wt_vif": (model_10_wt_vif, get_cov_model9_init_strategy, lambda x: int(x)),
    "model_10_wt_mock": (model_10_wt_mock, get_cov_model9_init_strategy, lambda x: int(x)),
    "model_10_vif_mock": (model_10_vif_mock, get_cov_model9_init_strategy, lambda x: int(x)),
    "model_10_wt_vif_mock": (model_10_wt_vif_mock, get_cov_model9_init_strategy, lambda x: int(x)),
    "model_11_path_length": (model_11_path_length, init_to_uniform, lambda x: int(x)),
        }

def model_dispatcher(model_name, model_data, save_dir):
    if model_name in _model_dispatch:
        model, init_strategy, data_func = _model_dispatch[model_name]
        model_data = data_func(model_data)
    elif model_name == "cov_model5":
        model_data = int(model_data)
        model = cov_model5
        init_strategy = get_cov_model4_init_strategy(model_data)
    elif model_name == "cov_model6":
        model_data = int(model_data)
        model = cov_model6
        init_strategy = get_cov_model4_init_strategy(model_data)
    elif model_name == "cov_model7":
        model_data = int(model_data)
        model = cov_model7
        init_strategy = get_cov_model7_init_strategy(model_data)
    elif model_name == "cov_model8":
        model_data = int(model_data)
        model = cov_model8
        init_strategy = get_cov_model4_init_strategy(model_data)
    elif model_name == "cov_model9":
        model_data = int(model_data)
        model = cov_model9
        init_strategy = get_cov_model9_init_strategy(model_data)
    elif model_name == "model_10_wt":
        model_data = int(model_data)
        model = model_10_wt
        init_strategy = get_cov_model9_init_strategy(model_data)
    elif model_name == "model12":
        model_data = None
        model = model12
        init_strategy = init_to_uniform
    elif model_name == "model13":
        model_data = None
        model = model13
        init_strategy = init_to_uniform
    elif model_name == "model14":
        print("model14 in main")
        if model_data is None:
            model_data = model14_data_getter()
        else:
            temp = model14_data_getter()
            temp["flattened_apms_similarity_scores"] = model_data
            model_data = temp
        # test line
        model = model14
        init_strategy = init_to_uniform
    elif model_name == "model15":
        print("model15 in main")
        if model_data is None:
            model_data = model14_data_getter()
        else:
            temp = model14_data_getter()
            temp["flattened_apms_similarity_scores"] = model_data
            model_data = temp
        # test line
        model = model15
        init_strategy = init_to_uniform
    elif model_name == "model22_ll_lp":
        model_data = model22_ll_lp_data_getter(save_dir)
        init_strategy = init_to_uniform
    elif model_name == "model23_ll_lp":
        model = model23_ll_lp
        model_data = model23_ll_lp_data_getter(save_dir)
        model_data = model23_data_transformer(model_data)
        #init_strategy = init_to_uniform
        #init_strategy = model_23_ll_lp_init_to_zero_strategy(model_data)
        init_strategy = model_23_ll_lp_init_to_data_strategy(model_data)
    else:
        raise ValueError(f"Invalid {model_name}")
    return model, model_data, init_strategy

def init_position_dispatcher(initial_position_fp, model_name):
    if model_name == "model14":
        with open(initial_position_fp, "rb") as f:
            d = pkl.load(f) # dictionary of sample sites
        return init_to_value(values=d)
    else:
        raise NotImplementedError 

def trace_model_shapes(model, arg):
    with numpyro.handlers.seed(rng_seed=1):
        trace = numpyro.handlers.trace(model).get_trace(arg)
    print(numpyro.util.format_shapes(trace))

@click.command()
@click.option("--model-id", type=str, help="identifier is prepended to saved files")
@click.option("--rseed", type=int, default=0)
@click.option("--model-name", type=str)
@click.option("--model-data")
@click.option("--num-warmup", type=int)
@click.option("--num-samples", type=int)
@click.option("--include-potential-energy", is_flag=True, default=False)
@click.option("--include-mean-accept-prob", is_flag=True, default=False)
@click.option("--include-extra-fields", is_flag=True, default=True) 
@click.option("--progress-bar", is_flag=True, default=False)
@click.option("--save-dir", default=None)
@click.option("--initial-position", default=None)
@click.option("--save-warmup", default=False)
@click.option("--load-warmup", default=False)
@click.option("--jax-profile", default=False, is_flag=True)
def main(model_id,
         rseed,
         model_name,
         model_data,
         num_warmup,
         num_samples,
         include_potential_energy,
         include_mean_accept_prob,
         indluce_extra_fields,
         progress_bar,
         save_dir,
         initial_position,
         save_warmup,
         load_warmup,
         jax_profile):
    _main(model_id = model_id,
          rseed = rseed,
          model_name = model_name,
          model_data = model_data,
          num_warmup = num_warmup,
          num_samples = num_samples,
          include_potential_energy = include_potential_energy,
          include_mean_accept_prob = include_mean_accept_prob,
          include_extra_fields = include_extra_fields,
          progress_bar = progress_bar,
          save_dir = save_dir,
          initial_position = initial_position,
          save_warmup = save_warmup,
          load_warmup = load_warmup,
          jax_profile = jax_profile)

def _main(model_id,
         rseed,
         model_name,
         model_data,
         num_warmup,
         num_samples,
         include_potential_energy,
         include_mean_accept_prob,
         include_extra_fields,
         progress_bar,
         save_dir,
         initial_position,
         save_warmup,
         load_warmup,
         jax_profile,):
    log_path = Path(save_dir) / "_model_variations.log"
    logging.basicConfig(filename=log_path, filemode='w')
    entered_main_time = time.time()
    rng_key = jax.random.PRNGKey(rseed)
    eprint(f"Model ID: {model_id}")
    eprint(f"Model Name: {model_name}")
    eprint(jax.devices())
    extra_fields = ()
    if include_extra_fields:
        extra_fields = ("potential_energy", # N
                        "diverging",      # N,
                        "accept_prob",   # N
                        "mean_accept_prob", #N
                        #"step_size",
                        #"inverse_mass_matrix",
                        #"effective_sample_size",
                        #"r_hat",
                        "num_steps", # N all same
                        "energy", # N
                        #"adapt_state",
                            # step_size N
                        #"trajectory_length",
        )
    model, model_data, init_strategy = model_dispatcher(model_name, model_data, save_dir)
    if include_potential_energy:
        extra_fields = extra_fields + ("potential_energy",)
    if include_mean_accept_prob:
        extra_fields = extra_fields + ("mean_accept_prob",)

    # Check if we should initalize to a certain value
    model_id = str(model_id)
    model_name = str(model_name)

    if initial_position:
        init_strategy = init_position_dispatcher(initial_position, model_name)
    warmup_savename = model_id + "_" + model_name + "_" + "hmc_warmup.pkl"
    warmup_savename = str(Path(save_dir) / warmup_savename)
    nuts = NUTS(model, init_strategy=init_strategy)
    # Do the warmup
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
    if save_warmup:
        rng_key, key = jax.random.split(rng_key)
        mcmc.warmup(key, model_data = model_data)
        warmup_state = mcmc.last_state
        with open(warmup_savename, "wb") as f:
            pkl.dump(warmup_state, f)
    if load_warmup:
        with open(warmup_savename, "rb") as f:
            warmup_state = pkl.load(f)
        mcmc.post_warmup_state = warmup_state 
    if jax_profile:
        profile_path = str(Path(save_dir) / "jax-trace")
        logging.info(f"writing trace to {profile_path}")
        with jax.profiler.trace(profile_path):
            mcmc.run(rng_key, model_data, extra_fields=extra_fields)
    else:
        mcmc.run(rng_key, model_data, extra_fields=extra_fields)
    finished_mcmc_time = time.time()
    elapsed_time = finished_mcmc_time - entered_main_time
    eprint(f"{num_samples} sampling steps complete")
    eprint(f"elapsed time (s): {elapsed_time}")
    eprint(f"elapsed time (m): {elapsed_time / 60}")
    eprint(f"elapsed time (h): {elapsed_time / (60 * 60)}")
    savename = model_id + "_" + model_name + "_" + str(rseed) + ".pkl"
    if save_dir is not None:
        savename = Path(save_dir) / savename
        savename = str(savename)
    fields = mcmc.get_extra_fields()
    samples = mcmc.get_samples()

    out = {"extra_fields": fields, "samples": samples, "elapsed_time": elapsed_time,
           }
    with open(savename, "wb") as file:
        pkl.dump(out, file) 
    if model_name == "model23_ll_lp":
        data_savename = model_id + "_" + model_name + "_" + str(rseed) + "_model_data.pkl"
        with open(data_savename, "wb") as file:
            pkl.dump(model_data, file)
if __name__ == "__main__":
    main()
