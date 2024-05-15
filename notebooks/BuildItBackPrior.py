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
import dev_clique_and_community_src as dcc
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

def u_from_edgelist_df(e):
    u = gbf.UndirectedEdgeList()
    u.update_from_df(
        e, a_colname='a', b_colname='b',
        edge_value_colname='w', multi_edge_value_merge_strategy='max')
    u.reindex(reindexer, enforce_coverage=False)
    return u
    

# -

saint_prior = get_saint_prior(r1)

saint_prior = saint_prior.sort_index(axis=0)
saint_prior = saint_prior.sort_index(axis=1)
r1.A_df.sort_index(axis=0, inplace=True)
r1.A_df.sort_index(axis=1, inplace=True)


def get_degree_prior_from_matrix_df(matrix_df):
    N, _ = matrix_df.shape
    degree = np.sum(matrix_df.values, axis=0).reshape((N, 1)) / N
    degree = degree * degree.T
    degree_prior = 1 - degree
    return pd.DataFrame(degree_prior, columns = matrix_df.columns, index=matrix_df.index)


sns.heatmap(get_degree_prior_from_matrix_df(r1.A_df) * r1.A_df)

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

degree_prior = get_degree_prior_from_matrix_df(r1.A_df)


def hist(ax, x):
    ax.hist(x, bins=100)
fig, ax = plt.subplots(6, 1)
tril_indices = np.tril_indices_from(r1.A_df, k=-1)
hist(ax[0], np.ravel(r1.A_df.values[tril_indices]))
ax[0].set_xlabel("Average edge score")
hist(ax[1], np.ravel(saint_prior.values[tril_indices]))
ax[1].set_xlabel("Max pair saint score")
hist(ax[2], np.ravel(pairwise_product.values[tril_indices]))
ax[2].set_xlabel("Pairwise product")
hist(ax[3], np.ravel(pairwise_average.values[tril_indices]))
ax[3].set_xlabel("Pairwise addition")
hist(ax[4], np.ravel(degree_prior.values[tril_indices]))
ax[4].set_xlabel("degree_prior")
hist(ax[5], np.ravel(degree_prior * pairwise_average))
ax[5].set_xlabel("deg_x_av")
plt.tight_layout()

m3 = (saint_prior + degree_prior * pairwise_average)
m3_edgelist_df = sa.matrix_df_to_edge_list_df(m3)
u_m3 = u_from_edgelist_df(m3_edgelist_df)

pairwise_product_edgelist_df = sa.matrix_df_to_edge_list_df(pairwise_product)

pairwise_average_edgelist_df = sa.matrix_df_to_edge_list_df(pairwise_average)

saint_prior_edgelist_df = sa.matrix_df_to_edge_list_df(saint_prior)

degree_prior_edgelist_df = sa.matrix_df_to_edge_list_df(degree_prior)

reindexer = gbf.get_cullin_reindexer()

u_degree_prior = u_from_edgelist_df(degree_prior_edgelist_df)

degree_prior_x_pair_edgelist_df = sa.matrix_df_to_edge_list_df(degree_prior * pairwise_product)
u_degree_prior_x_pair_prod = u_from_edgelist_df(degree_prior_x_pair_edgelist_df)

degree_prior_x_pair_av_edgelist_df = sa.matrix_df_to_edge_list_df((degree_prior * pairwise_average))
u_degree_prior_x_pair_av = u_from_edgelist_df(degree_prior_x_pair_av_edgelist_df)

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
    degree_prior = u_degree_prior,
    deg_x_prod = u_degree_prior_x_pair_prod,
    deg_x_av = u_degree_prior_x_pair_av,
    m3 = u_m3,
)

references = dict(
  costructure = gbf.get_pdb_ppi_predict_cocomplex_reference(),
  direct = gbf.get_pdb_ppi_predict_direct_reference()
)

importlib.reload(gbf)

gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", "costructure"),
    ("average_edge", "costructure"),
    ("pair_prod", "costructure"),
    ("pair_av", "costructure"),
    ("degree_prior", "costructure"),
    ("deg_x_prod", "costructure"),
    ("deg_x_av", "costructure"),
    #("m3", "costructure",)
    ),
    multi_save_suffix="_co_structure")

predictions['m3']

gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", "direct"),
    ("average_edge", "direct"),
    ("pair_prod", "direct"),
    ("pair_av", "direct"),
    ("degree_prior", "direct"),
    ("deg_x_prod", "direct"),
    ("deg_x_av", "direct"),
    #("m3", "direct"),
    ),
    multi_save_suffix="_direct")



# +
# Let's say we want to predict a network based on the ROC type curve
# 1% of the total interaction space
# -

pp_top_node_set, pp_top_edge_set = dcc.get_top_edges_upto_threshold(pairwise_product_edgelist_df, threshold=0.9997)

deg_x_av_top_node_set, deg_x_av_top_edge_set = dcc.get_top_edges_upto_threshold(
    degree_prior_x_av_edgelist_df, threshold=0.9)


# +
def fraction_of_all_pairs(n_nodes, nedges):
    all_pairs = math.comb(n_nodes, 2)
    return nedges / all_pairs

f(len(pp_top_node_set), len(pp_top_edge_set))
# -

len(pp_top_node_set), len(pp_top_edge_set), (len(pp_top_edge_set) / 27261) * 100

len(deg_x_av_top_node_set), len(deg_x_av_top_edge_set), (len(deg_x_av_top_edge_set) / 27261) * 100

# +
# Build a network using the top edges
# -

net_pp = dcc.pyvis_plot_network(pp_top_edge_set)

# +
#net_pp.show("pp.html")
# -

net_deg_x_av = dcc.pyvis_plot_network(deg_x_av_top_edge_set)

net_deg_x_av.show("deg_x_av.html")

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
