# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle as pkl
import xarray as xr
from numpyro.distributions.constraints import Constraint
import pandas as pd
from pathlib import Path
import math
import scipy as sp
# try and fit the data to a model 

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
import jax
from numpyro.contrib.funsor import config_enumerate, infer_discrete
from numpyro.infer import NUTS, MCMC
import jax.numpy as jnp
import arviz as az
import numpyro.distributions as dist
from jax.tree_util import Partial
import sympy as sympy
import jax.scipy as jsp
import inspect
from collections import namedtuple
import blackjax
import blackjax.smc.resampling as resampling
from numpyro.distributions.transforms import Transform
from numpyro.distributions import constraints
from jax.tree_util import tree_flatten, tree_unflatten

from functools import partial
# -

import _model_variations as mv

# Notebook Bools
CALCULATE_DISTANCE_REFERENCE = False
SAVE_DISTANCE_REFERENCE = False

with open("direct_benchmark.pkl", 'rb') as f:
    direct_benchmark = pkl.load(f)
reference_matrix = direct_benchmark.reference.matrix

if CALCULATE_DISTANCE_REFERENCE:
    D_reference = mv.shortest_paths_up_to_N(direct_benchmark.reference.matrix.values, 25)
    D_reference = xr.DataArray(D_reference, coords=reference_matrix.coords)
else:
    D_reference = mv.load("direct_reference_distance_matrix_up_to_25.pkl")

if SAVE_DISTANCE_REFERENCE:
    with open("direct_reference_distance_matrix_up_to_25.pkl", "wb") as f:
        pkl.dump(D_reference, f)

# Correct D_reference so that self distances are 0
N, _ = D_reference.shape
temp = D_reference.values
diag = np.diag_indices(N)
#temp[np.diag_indices(N)] = 0 
temp[diag] = 0
D_reference = xr.DataArray(temp, coords=D_reference.coords)

D_reference

plt.matshow(temp)

apms_pearson_r = mv.load("xr_apms_correlation_matrix.pkl")

assert apms_pearson_r.shape == D_reference.shape

# Id Mapping
reference_dict = mv.preyu2uid_mapping_dict()
uid2gene_name = {val: key for key,val in reference_dict.items()}
gene_ids = [uid2gene_name[i] for i in apms_pearson_r.uid_preyu.values]
apms_pearson_r = xr.DataArray(apms_pearson_r.values, coords={"preyu": gene_ids, "preyv": gene_ids})

plt.matshow(D_reference.values)

assert np.alltrue(D_reference.preyu == apms_pearson_r.preyu)
D_flat = mv.matrix2flat(D_reference.values)
pearson_r_flat = mv.matrix2flat(apms_pearson_r.values)

# +
D_flat = np.array(D_flat)
pearson_r_flat = np.array(pearson_r_flat)

fig, ax = plt.subplots(1, 2)
ax[0].hist(D_flat)
ax[1].hist(D_flat, range=(0, 19), bins=20)
ax[0].set_xlabel("Taxicab distance")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Taxicab distance")
plt.tight_layout()
# -

plt.plot(pearson_r_flat, D_flat, 'k.')
plt.xlabel("Pearson R")
plt.ylabel("Distance")

D_at_small_correlations = D_flat[pearson_r_flat <= 0.6]
D_at_large_correlations = D_flat[pearson_r_flat > 0.6]

# +
len(pearson_r_flat) % 2
#middle_index = len(pearson_r_flat) // 2
middle_index = int(len(pearson_r_flat) * 0.1)

small_D = D_flat[0:middle_index]
large_D = D_flat[middle_index:-1]

fig, ax = plt.subplots(1, 2)
ax[0].hist(small_D, label="Small", alpha=0.5, density=True)
ax[0].hist(large_D, label="Large", alpha=0.5, density=True)

ax[1].hist(small_D, range=(0, 19), alpha=0.5, bins=20, density=True)
ax[1].hist(large_D, range=(0, 19), alpha=0.5, bins=20, density=True)
ax[0].set_xlabel("Taxicab distance")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Taxicab distance")
ax[0].legend()
plt.tight_layout()

# +
D_at_small_correlations = D_flat[pearson_r_flat <= 0.6]
D_at_large_correlations = D_flat[pearson_r_flat > 0.6]
ax[0].hist(D_flat)
ax[1].hist(D_flat, range=(0, 19), bins=20)
ax[0].set_xlabel("Taxicab distance")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Taxicab distance")
plt.tight_layout()

plt.hist(D_at_large_correlations, label="D_at_small_correlations", alpha=0.5)
plt.hist(D_at_small_correlations, label="D_at_large_correlations", alpha=0.5)


plt.legend()
# -



plt.plot(pearson_r_flat > 0.6, D_flat, 'k.')
plt.xlabel("Pearson R")
plt.ylabel("Distance")

N, _ = D_reference.shape
M = math.comb(N, 2)
assert len(D_flat) == M

fig, ax = plt.subplots(1, 2)
ax[0].matshow(apms_pearson_r.values)
ax[1].matshow(D_reference)

# +
mini_sel = ["ELOB", "ELOC", "vifprotein", "PEBB",
            "CUL5", "CUL2", "RBX1", "RBX2", "NEDD8", "ARI1", "ARI2"
            ]

apms_mini = apms_pearson_r.sel(preyu=mini_sel, preyv=mini_sel)
D_mini = D_reference.sel(preyu=mini_sel, preyv=mini_sel)
#D_mini[np.diag_indices_from(D_mini.values)] = 0
tril_indices = np.tril_indices_from(apms_mini, k=-1)

# +
fig, ax = plt.subplots(1, 2)#, figsize=(8, 8))
for a in ax.flat:
    a.set_aspect("equal", "box")

mappable = ax[0].matshow(D_mini, cmap="binary", vmin=0, vmax=5)
plt.colorbar(mappable, location="bottom")
mappable = ax[1].matshow(apms_mini, cmap="coolwarm", vmin=-1, vmax=1)
ax[0].set_yticks(np.arange(len(D_mini)), labels=mini_sel)
#ax[0].set_xticks(np.arange(len(D_mini)), labels=mini_sel)
plt.tight_layout()
plt.colorbar(mappable, location="bottom")
ax[0].set_xlabel("Distance")
ax[1].set_xlabel("Profile Similarity (R)")
plt.show()


# +
def plot_expected_correlation_decay():
    """
    Rank prey by max saint score
    
    
    
    """
    ...
    
def get_max_saint_scores(df_new, apms_pearson_r):
    max_saint_score = {k: 0 for k in apms_pearson_r.preyu.values}

    for label, r in df_new.iterrows():
        if label in max_saint_score:
            val = max_saint_score[label]
            saint_score = r['SaintScore']
            if saint_score > val:
                max_saint_score[label] = saint_score
    
    coordinate = np.zeros(len(max_saint_score), dtype="U14")
    scores = np.zeros(len(max_saint_score))
    for i, (key, value) in enumerate(max_saint_score.items()):
        coordinate[i] = key
        scores[i] = value
    return xr.DataArray(scores, coords={"preyu": coordinate})


# -

max_saint_xr = get_max_saint_scores(df_new, apms_pearson_r)
max_saint_xr = xr.DataArray(max_saint_xr.to_pandas().sort_values())

plt.hist(max_saint_xr.values, bins=len(max_saint_xr) // 8)
plt.ylabel("Frequency")
plt.xlabel("Max Saint Score")
plt.show()

top_prey = max_saint_xr.where(max_saint_xr == 1, drop=True)


# +
def matrix_regplot(matrix_a, matrix_b, max_distance=10, ylim=5):
    tril_indices = np.tril_indices_from(matrix_a, k=-1)
    a = matrix_a[tril_indices]
    b = matrix_b[tril_indices]
    plt.subplot(121)

    #a = apms_mini.values[tril_indices]
    #b = D_mini.values[tril_indices]
    sel = b < max_distance
    linear_regression = sp.stats.linregress(a[sel], b[sel], alternative='less')
    
    plt.plot(a, b, 'k.')
    plt.subplot(122)
    plt.plot(a, b, 'k.')
    plt.title(f"r: {linear_regression.rvalue:.2f} p: {linear_regression.pvalue:.8f}")


    x = np.arange(-1, 1, 0.01)
    plt.plot(x, linear_regression.slope * x + linear_regression.intercept)

    plt.xlabel("AP-MS Profile Similarity (pearson R)")
    plt.ylabel("Network Distance")
    results = sp.stats.pearsonr(a[sel], b[sel])
    plt.xlim(-1, 1)
    plt.ylim(0, ylim)
    plt.show()
    
matrix_regplot(apms_mini.values, D_mini.values)
# -

sel = top_prey.preyu.values
matrix_regplot(apms_pearson_r.sel(preyu=sel, preyv=sel).values,
              D_reference.sel(preyu=sel, preyv=sel).values, ylim=15)


# +
def calculate_network_distance_score_correlation(d, s, is_xr=True, max_distance=10):
    """
    d : a distance matrix xarray
    s : a score matrix xarray
    """
    
    if is_xr:
        d = d.values
        s = s.values
    assert d.shape == s.shape, (d.shape, s.shape)
    tril_indices = np.tril_indices_from(d, k=-1)
    d = d[tril_indices]
    s = s[tril_indices]
    
    sel = d < max_distance
    linear_regression = sp.stats.linregress(d[sel], s[sel], alternative='less')
    return linear_regression



def get_shortest_distance_from_any_bait(D_reference, bait_names = None):
    """
    
    """
    if bait_names is None:
        bait_names = ["CUL5", "PEBB", "ELOB"]
    bait_matrix = D_reference.sel(preyv=bait_names)
    return bait_matrix.min(axis=1)
    


# -

min_distance_to_a_bait = get_shortest_distance_from_any_bait(D_reference)

plt.hist(min_distance_to_a_bait, bins=50, range=(0,15))
plt.xlabel("Minimal taxicab distance to a bait")
plt.ylabel("Frequency")
plt.title(f"Close {np.sum(min_distance_to_a_bait.values < 15)}, Far {np.sum(min_distance_to_a_bait.values >= 15)}")
plt.show()

plt.plot(min_distance_to_a_bait, max_saint_xr, 'k.')
plt.ylabel("Saint Score")
plt.xlabel("Min Distance to Bait")

min_distance_to_a_bait

max_saint_xr

bait_matrix = get_shortest_distance_from_any_bait(D_reference)

calculate_network_distance_score_correlation(D_mini, apms_mini)


def correlation_decohearence(apms_pair_score, D_reference, stepsize=10,
                            max_distance=10):
    # Sort saint scores in ascending order
    #max_saint_scores = xr.DataArray(max_saint_scores.to_pandas().sort_values())
    #for i in range(len(max_saint_xr), 10):
    
    min_distance_from_any_bait = get_shortest_distance_from_any_bait(D_reference)
    min_distance_from_any_bait = xr.DataArray(min_distance_from_any_bait.to_pandas().sort_values())
    scores = min_distance_from_any_bait
    
    
    # Select top and bottom N saint scores
    factor = 1
    N = len(scores) // (factor * stepsize)
    columns=["N", "slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
    output = np.zeros((N, len(columns)))
    top_scores = pd.DataFrame(output.copy(), columns=columns)
    bot_scores = pd.DataFrame(output.copy(), columns=columns)

    k=0
    f = partial(calculate_network_distance_score_correlation, max_distance=max_distance)
    for i in range(stepsize, len(scores) // factor, stepsize):
        
        bot_N = scores[0:i]
        top_N = scores[-i:]
        
        bot_prey_sel = bot_N.preyu.values
        top_prey_sel = top_N.preyu.values
        
        top_apms = apms_pair_score.sel(preyu=top_prey_sel, preyv=top_prey_sel)
        bot_apms = apms_pair_score.sel(preyu=bot_prey_sel, preyv=bot_prey_sel)
        
        top_D_ref = D_reference.sel(preyu=top_prey_sel, preyv=top_prey_sel)
        bot_D_ref = D_reference.sel(preyu=bot_prey_sel, preyv=bot_prey_sel)
        
        
        top_reg = f(top_apms, top_D_ref)._asdict()
        bot_reg = f(bot_apms, bot_D_ref)._asdict()
        
        top_reg = pd.Series({"N": i} | top_reg)
        bot_reg = pd.Series({"N": i} | bot_reg)
        
        top_scores.iloc[k, :] = top_reg
        bot_scores.iloc[k, :] = bot_reg
        k += 1
        #if k > 2:
        #    break
    return top_scores, bot_scores
        

min_distance_to_a_bait

t, b = correlation_decohearence(apms_pearson_r, D_reference, stepsize=5, max_distance=15)

t

b

# +
# Drop the last row
#t = t.drop(600)
#b = b.drop(600)

plt.subplot(221)
plt.plot(t['N'], t['rvalue'], label="Far from any bait")
plt.plot(b['N'], b['rvalue'], label="close to any bait")
plt.legend()
#plt.xlabel("N")
plt.ylabel("R value")
plt.subplot(222)
plt.plot(t['N'], -np.log10(t['pvalue']))
plt.plot(t['N'], -np.log10(b['pvalue']))
#plt.xlim(0,50)
#plt.ylim(0, 15)
plt.ylabel("- log p-value")
#plt.hlines(5, 0, 1500, 'r')
#plt.ylim(0, 0.0000000000001)

plt.subplot(223)
plt.plot(t['N'], t['slope'])
plt.plot(t['N'], b['slope'])
plt.xlabel("N")
plt.ylabel("slope")
plt.tight_layout()
plt.subplot(224)
plt.plot(t['N'], t['intercept'])
plt.plot(t['N'], b['intercept'])
plt.ylabel("Intercept")
plt.xlabel("N")
plt.show()

# +
# Do a scatter plot of minimal bait distance to spectral count variation
# -

# Conclusions and Implications for Modeling
# - We know shortest bait prey paths for 546 (18%) nodes and don't know for 2459 (81%) nodes
# - Thus 148, 785 edges are supported by at least one PDB bait prey path
# - 3,022,111 edges are not supported by any bait prey path
# - Effects that add to decohearance are
#     1. Missing information in the PDB
#     2. type II annotation errors
#     3. variation in AP-MS data
# - At close network distances, profile similarity is negativley correlated with network distances
# - What are the top ranked edges with no support in the PDB?
# - Do I trust the SAINT score?

math.comb(2459, 2)

t.drop(600)

# +

df = pd.DataFrame(np.zeros((4, 6)), columns=list(t._asdict().keys()))
# -

top_saint

b



df + pd.Series(t._asdict())

pd.Series(t._asdict())

correlation_decohearence(apms_pearson_r, D_reference, max_saint_xr)



max_saint_xr

correlation_decohearence(apms_pearson_r, D_reference, max_saint_xr)

max_saint_xr.where(max_saint_xr < )

xr.DataArray(max_saint_xr.to_pandas().sort_values())

max_saint_xr.sort_values()

help(max_saint_xr.sortby)

D_reference.sel(preyu=top_prey.preyu.values)

df_new = mv.load("df_new.pkl")

plt.colorbar(mappable, location="bottom")
ax[2].plot(apms_mini[tril_indices], D_mini[tril_indices], 'k.')
ax[2].set_xlabel("Pearson R")
ax[2].set_ylabel("Network Distance")
ax[2].set_xlim(-1, 1)
#ax[2].set_aspect("equal", "box")
plt.tight_layout()
plt.show()
#plt.colorbar(mappable, cax=ax[2])

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].matshow(D_mini, cmap="binary")
plt.subplot(122)
plt.xticks(np.arange(len(D_mini)), labels=mini_sel)
plt.yticks(np.arange(len(D_mini)), labels=mini_sel)

fig, ax = plt.subplots(1, 2)
ax1 = ax[0]
ax[0].matshow(apms_mini > 0.6)
ax[0].set_yticklabels(mini_sel)
#ax1.set_yticklabels(mini_sel)
ax[1].matshow(D_mini)
ax[0].set_title("AP-MS R")
ax[1].set_title("D")

fig, axs = plt.subplots(1, 2, figsize=(8, 8))
apms_mini.plot(ax=axs[0])
D_mini.plot(ax=axs[1], cmap="binary")
plt.tight_layout()

help(plt.tight_layout)

plt.matshow(apms_mini > 0.4)


mixed = np.tril(D_reference.values, k=-1)


plt.plot(np.ravel(apms_pearson_r.values), np.ravel(D_reference.values), 'k.', alpha=0.1)
plt.show()
