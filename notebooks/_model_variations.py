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

def matrix2flat(M, row_major=True):
    """
    An iterative method to flatten a matrix
    """
    n, m = M.shape
    N = math.comb(n, 2)
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
            "null_dist" : null_dist} 
    
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
    ntest = model_data['ntest']
    flattened_apms_similarity_scores = flattened_apms_similarity_scores[0:ntest]
    shuffled = model_data["flattened_apms_shuffled_similarity_scores"][0:ntest]
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

@click.command()
@click.option("--model-id", type=str, help="identifier is prepended to saved files")
@click.option("--rseed", type=int, default=0)
@click.option("--model-name", type=str)
@click.option("--model-data")
@click.option("--num-warmup", type=int)
@click.option("--num-samples", type=int)
@click.option("--include-potential-energy", is_flag=True, default=False)
@click.option("--include-mean-accept-prob", is_flag=True, default=False)
@click.option("--progress-bar", is_flag=True, default=False)
def main(model_id, rseed, model_name,model_data, num_warmup, num_samples, include_potential_energy, 
    progress_bar, include_mean_accept_prob):
    entered_main_time = time.time()
    rng_key = jax.random.PRNGKey(rseed)
    eprint(f"Model ID: {model_id}")
    eprint(f"Model Name: {model_name}")
    eprint(jax.devices())
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
        model_data = model14_data_getter()
        # test line
        model_data['ntest'] = 100
        model = model14
        init_strategy = init_to_uniform
    else:
        raise ValueError(f"Invalid {model_name}")
    extra_fields = ()
    if include_potential_energy:
        extra_fields = extra_fields + ("potential_energy",)
    if include_mean_accept_prob:
        extra_fields = extra_fields + ("mean_accept_prob",)
    nuts = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
    mcmc.run(rng_key, model_data, extra_fields=extra_fields)
    finished_mcmc_time = time.time()
    elapsed_time = finished_mcmc_time - entered_main_time
    eprint(f"{num_samples} sampling steps complete")
    eprint(f"elapsed time (s): {elapsed_time}")
    eprint(f"elapsed time (m): {elapsed_time / 60}")
    eprint(f"elapsed time (h): {elapsed_time / (60 * 60)}")
    savename = model_id + "_" + model_name + "_" + str(rseed) + ".pkl"
    fields = mcmc.get_extra_fields()
    samples = mcmc.get_samples()
    out = {"extra_fields": fields, "samples": samples, "elapsed_time": elapsed_time}
    with open(savename, "wb") as file:
        pkl.dump(out, file) 

if __name__ == "__main__":
    main()

#model_id, rseed, model_name,
#         model_data, num_warmup, num_samples, include_potential_energy
