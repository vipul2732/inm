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

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import graph_tool
import pandas as pd
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
from pathlib import Path
import math
import generate_benchmark_figures as gbf
import timeit
import networkx as nx
import _model_variations as mv
import pickle as pkl
from collections import namedtuple
import pyvis
import sampling_assessment as sa
from collections import defaultdict
import importlib
importlib.reload(gbf)
# %matplotlib inline



# +
def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)
    

# -

sa._e1

# Low prior mock uniform 20k
r1 = sa.get_results(sa._e1)

# +
# Minimal saint scoring


# 1. Calculate maximal composite prey scores
def calculate_maximal_prey_saint_score_dict(results_path):
    composite_table = pd.read_csv(Path(sa._e1) / "composite_table.tsv", sep="\t")
    
    t = defaultdict(int)
    for i, r in composite_table.iterrows():
        prey_name = r['Prey']
        val = r['MSscore']
        if t[prey_name] < val:
            t[prey_name] = val
    return t

def calculate_pairwise_maximal_saint(max_saint_score_dict):
    """
    The prior probability of an edge given two SAINT scores
    p(co-purifying| A), p(co-purifying| B)
    """
    t = max_saint_score_dict
    n = len(t)
    matrix = np.zeros((n, n))
    k_list = [(name, max_score) for name, max_score in t.items()]
    names = [h[0] for h in k_list]
    for i in range(n):
        for j in range(1, i):
            a = k_list[i][1]
            b = k_list[j][1]
            matrix[i, j] = a * b
    matrix = matrix + matrix.T
    return pd.DataFrame(matrix, columns=names, index=names)

def get_saint_prior(r):
    # get the maximal prey scores
    max_prey_scores = calculate_maximal_prey_saint_score_dict(r.path)
    
    # filter by nodes in network
    temp = {}
    for key, value in max_prey_scores.items():
        if key in r.model_data['name2node_idx']:
            temp[key] = value
    max_prey_scores = temp
    return calculate_pairwise_maximal_saint(max_prey_scores)



# -

saint_prior = get_saint_prior(r1)

saint_prior = saint_prior.sort_index(axis=0)
saint_prior = saint_prior.sort_index(axis=1)
r1.A_df.sort_index(axis=0, inplace=True)
r1.A_df.sort_index(axis=1, inplace=True)

sns.heatmap(r1.A_df)

sns.heatmap(saint_prior)

# +
assert np.alltrue(saint_prior.columns == r1.A_df.columns)
assert np.alltrue(saint_prior.index == r1.A_df.index)
assert np.alltrue(saint_prior.index == r1.A_df.columns)

pairwise_product = r1.A_df * saint_prior
# -

pairwise_average = (r1.A_df + saint_prior) / 2

sns.heatmap(pairwise_product)

sns.heatmap(pairwise_average)


def hist(ax, x):
    ax.hist(x, bins=100)
fig, ax = plt.subplots(4, 1)
tril_indices = np.tril_indices_from(r1.A_df, k=-1)
hist(ax[0], np.ravel(r1.A_df.values[tril_indices]))
ax[0].set_xlabel("Average edge score")
hist(ax[1], np.ravel(saint_prior.values[tril_indices]))
ax[1].set_xlabel("Max pair saint score")
hist(ax[2], np.ravel(pairwise_product.values[tril_indices]))
ax[2].set_xlabel("Pairwise product")
hist(ax[3], np.ravel(pairwise_average.values[tril_indices]))
ax[3].set_xlabel("Pairwise addition")
plt.tight_layout()

pairwise_product_edgelist_df = sa.matrix_df_to_edge_list_df(pairwise_product)

pairwise_average_edgelist_df = sa.matrix_df_to_edge_list_df(pairwise_average)

saint_prior_edgelist_df = sa.matrix_df_to_edge_list_df(saint_prior)

reindexer = gbf.get_cullin_reindexer()

u = gbf.UndirectedEdgeList()
u.update_from_df(
    pairwise_average_edgelist_df, a_colname='a', b_colname='b',
    edge_value_colname='w', multi_edge_value_merge_strategy='max')
u_pairwise_average = u
u_pairwise_average.reindex(reindexer, enforce_coverage=False)

u = gbf.UndirectedEdgeList()
u.update_from_df(
    saint_prior_edgelist_df, a_colname='a', b_colname='b',
    edge_value_colname='w', multi_edge_value_merge_strategy='max')
u_saint_prior = u
u_saint_prior.reindex(reindexer, enforce_coverage=False)

u_saint_prior.node_intersection(gbf.get_pdb_ppi_predict_cocomplex_reference())

u_saint_prior

# +
u = gbf.UndirectedEdgeList()
u.update_from_df(pairwise_product_edgelist_df, a_colname='a', b_colname='b',
        edge_value_colname='w', multi_edge_value_merge_strategy='max')

u_pairwise = u
u_pairwise.reindex(reindexer, enforce_coverage=False)
# -

u_pairwise.node_intersection(r1.u)

predictions = dict(
    saint_prior = u_saint_prior,
    pair_prod = u_pairwise,
    average_edge = r1.u,
    pair_av = u_pairwise_average,
)

references = dict(
  pdb_costructure = gbf.get_pdb_ppi_predict_cocomplex_reference(),
  pdb_direct = gbf.get_pdb_ppi_predict_direct_reference()
)

importlib.reload(gbf)
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", "pdb_costructure"),
    ("saint_prior", "pdb_direct"),
    ("average_edge", "pdb_costructure"),
    ("average_edge", "pdb_direct"),
    ("pair_prod", "pdb_costructure"),
    ("pair_prod", "pdb_direct"),
    ("pair_av", "pdb_direct"),
    ("pair_av", "pdb_costructure")
    ))

# +
# Let's say we want to predict a network based on the ROC type curve
# 1% of the total interaction space

# +
# Build a network using the top edges
# -

np.sum(pairwise_product_edgelist_df['w'] >= 0.9995)

pairwise_product_edgelist_df.sort_values('w', ascending=False).iloc[0:50, :]

pairwise_product_edgelist_df.sort_values('w', ascending=False).iloc[0:50]

d = pd.read_csv("../significant_cifs/BSASA_concat.csv")

for i, r in d.iterrows():
    try:
        float(r['SASA12'])
        float(r['BSASA'])
        float(r['SASA1'])
    except:
        break

# +
# Filter out KeyErrors
sel1 = []
sel2 = []
sel3 = []
sel4 = []

def keyerror_append(x, v):
    if v != "KeyError":
        x.append(False)
    else:
        x.append(True)
for i, r in d.iterrows():
    keyerror_append(sel1, r['BSASA'])
    keyerror_append(sel2, r['SASA12'])
    keyerror_append(sel3, r['SASA1'])
    keyerror_append(sel4, r['SASA2'])

sel1 = np.array(sel1)
sel2 = np.array(sel2)
sel3 = np.array(sel3)
sel4 = np.array(sel4)

sel = sel1 & sel2
sel = sel & sel3
sel = sel & sel4
# -

# 44 rows are key error
d_filt = d[~sel]


def update_by_key(d, key):
    d.loc[:, key] = np.array(d[key].values, dtype=np.float64)
    return d


d_filt = update_by_key(d_filt, "BSASA")
d_filt = update_by_key(d_filt, "SASA12")
d_filt = update_by_key(d_filt, "SASA1")
d_filt = update_by_key(d_filt, "SASA2")

d_filt.sum()

sel = [isinstance(i, str) for i in d["SASA12"]]

d.loc[0, "SASA12"]

d.loc[~np.array(sel)]
