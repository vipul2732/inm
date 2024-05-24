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
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import graph_tool
import pandas as pd
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpyro.distributions as dist
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

def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)

print(sa._e1)
# -

# Low prior mock uniform 20k
m_erlp1u20k = sa.get_results(sa._e1)

w_erlp1u20k = sa.get_results("../results/se_sr_low_prior_1_wt_20k/")

r_diag = sa.get_results("../results/se_sr_low_prior_1_uniform_mock_2k_diagnose/se_sr_low_prior_1_uniform_mock_2k_diagnose_rseed_0/", rseed=0)

# +
#v_erlp1uv20k = sa.get_results("../results/se_sr_low_prior_1_uniform_")

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
    
def sort_matrix_df(mdf):
    mdf.sort_index(axis=0, inplace=True)
    mdf.sort_index(axis=1, inplace=True)
    
def get_degree_prior_from_matrix_df(matrix_df):
    N, _ = matrix_df.shape
    degree = np.sum(matrix_df.values, axis=0).reshape((N, 1)) / N
    degree = degree * degree.T
    degree_prior = 1 - degree
    return pd.DataFrame(degree_prior, columns = matrix_df.columns, index=matrix_df.index)

def plot_plotlist(plot_list):
    nrows = len(plot_list)
    fig, ax = plt.subplots(nrows, 1, figsize=(12, 10))
    tril_indices = np.tril_indices_from(m_erlp1u20k.A_df, k=-1)
    for i in range(nrows):
        hist(ax[i], np.ravel(plot_list[i][1].values[tril_indices]))
        ax[i].set_xlabel(plot_list[i][0])

    plt.tight_layout()

def hist(ax, x):
    ax.hist(x, bins=100, range=(0,1))


# -

reindexer = gbf.get_cullin_reindexer()

# +
m_saint_prior = get_saint_prior(m_erlp1u20k)

w_saint_prior = get_saint_prior(w_erlp1u20k)

m_degree_prior = get_degree_prior_from_matrix_df(m_erlp1u20k.A_df)

w_degree_prior = get_degree_prior_from_matrix_df(w_erlp1u20k.A_df)

# +
sort_matrix_df(m_saint_prior)
sort_matrix_df(w_saint_prior)

sort_matrix_df(m_erlp1u20k.A_df)
sort_matrix_df(w_erlp1u20k.A_df)

sort_matrix_df(m_degree_prior)
sort_matrix_df(w_degree_prior)
# -

m_pairwise_product = m_erlp1u20k.A_df * m_saint_prior # element-wise product
w_pairwise_product = w_erlp1u20k.A_df * w_saint_prior # element-wise product

m_pairwise_average = (m_erlp1u20k.A_df + m_saint_prior) / 2
w_pairwise_average = (w_erlp1u20k.A_df + w_saint_prior) / 2

m_deg_x_av = m_degree_prior * m_pairwise_average
w_deg_x_av = w_degree_prior * w_pairwise_average

m_deg_x_prod = m_degree_prior * m_pairwise_product
w_deg_x_prod = w_degree_prior * w_pairwise_product

m_corr = m_erlp1u20k.model_data["corr"]
m_corr.sort_index(axis=0, inplace=True)
m_corr.sort_index(axis=1, inplace=True)

# +
assert np.alltrue(m_saint_prior.columns == m_erlp1u20k.A_df.columns)
assert np.alltrue(m_saint_prior.index == m_erlp1u20k.A_df.index)
assert np.alltrue(m_saint_prior.index == m_erlp1u20k.A_df.columns)

assert np.alltrue(w_saint_prior.columns == w_erlp1u20k.A_df.columns)
assert np.alltrue(w_saint_prior.index == w_erlp1u20k.A_df.index)
assert np.alltrue(w_saint_prior.index == w_erlp1u20k.A_df.columns)

assert np.alltrue(m_pairwise_product.columns == m_erlp1u20k.A_df.columns)
assert np.alltrue(m_pairwise_average.columns == m_erlp1u20k.A_df.columns)
assert np.alltrue(m_corr.columns == m_saint_prior.columns)
# -

## Uncomment to plot various networks in matrix form
h = lambda x: sns.heatmap(x, vmin=0, vmax=1)
h(m_corr)
#h(m_erlp1u20k.A_df)
#h(w_erlp1u20k.A_df)
#h(m_degree_prior * m_erlp1u20k.A_df)
#h(w_degree_prior * w_erlp1u20k.A_df)
#h(m_degree_prior)
#h(w_degree_prior)
#h(m_saint_prior)
#h(w_saint_prior)
#h(m_pairwise_product)
#h(w_pairwise_product)
#h(m_pairwise_average)
#h(w_pairwise_average)
#h(m_deg_x_av)
#h(w_deg_x_av)
#del h

# +
m_plot_list = [
    ("m average edge score", m_erlp1u20k.A_df),
    ("m max pair saint score", m_saint_prior),
    ("m degree score", m_degree_prior),
    ("m pairwise product (edge & saint)", m_pairwise_product),
    ("m pairwise av (edge & saint)", m_pairwise_average),
    ("m deg_x_av", m_degree_prior * m_pairwise_average),
    ("m deg_x_prod", m_degree_prior * m_pairwise_product),
    ("m deg dist", (m_degree_prior-1) * -1)
    ]
w_plot_list = [
    ("w average edge score", w_erlp1u20k.A_df),
    ("w max pair saing score", w_saint_prior),
    ("w degree score", w_degree_prior),
    ("w pairwise product (edge & saint)", w_pairwise_product),
    ("w pairwise av (edge & saint)", w_pairwise_average),
    ("w deg_x_av", w_degree_prior * w_pairwise_average),
    ("w deg_x_prod", w_degree_prior * w_pairwise_product)    
]

plot_plotlist(m_plot_list)
    
# -

plot_plotlist(w_plot_list)

del plot_list

# +
m_pairwise_product_edgelist_df = sa.matrix_df_to_edge_list_df(m_pairwise_product)
w_pairwise_product_edgelist_df = sa.matrix_df_to_edge_list_df(w_pairwise_product)

m_pairwise_average_edgelist_df = sa.matrix_df_to_edge_list_df(m_pairwise_average)
w_pairwise_average_edgelist_df = sa.matrix_df_to_edge_list_df(w_pairwise_average)

m_saint_prior_edgelist_df = sa.matrix_df_to_edge_list_df(m_saint_prior)
w_saint_prior_edgelist_df = sa.matrix_df_to_edge_list_df(w_saint_prior)

m_degree_prior_edgelist_df = sa.matrix_df_to_edge_list_df(m_degree_prior)
w_degree_prior_edgelist_df = sa.matrix_df_to_edge_list_df(w_degree_prior)

m_deg_x_prod_edgelist_df = sa.matrix_df_to_edge_list_df(m_deg_x_prod)
w_deg_x_prod_edgelist_df = sa.matrix_df_to_edge_list_df(w_deg_x_av)

m_degree_prior_x_pair_edgelist_df = sa.matrix_df_to_edge_list_df(m_deg_x_prod)
w_degree_prior_x_pair_edgelist_df = sa.matrix_df_to_edge_list_df(w_deg_x_prod)

m_corr_edgelist_df = sa.matrix_df_to_edge_list_df(m_corr)
# -

u_m_pairwise_product = u_from_edgelist_df(m_pairwise_product_edgelist_df)
u_w_pairwise_product = u_from_edgelist_df(w_pairwise_product_edgelist_df)

u_m_corr = u_from_edgelist_df(m_corr_edgelist_df)

u_m_degree_prior = u_from_edgelist_df(m_degree_prior_edgelist_df)
u_w_degree_prior = u_from_edgelist_df(m_degree_prior_edgelist_df)

m_degree_prior_x_pair_edgelist_df = sa.matrix_df_to_edge_list_df(m_degree_prior * m_pa)
u_degree_prior_x_pair_prod = u_from_edgelist_df(degree_prior_x_pair_edgelist_df)

m_degree_prior_x_pair_av_edgelist_df = sa.matrix_df_to_edge_list_df((m_degree_prior * m_pairwise_average))
u_m_degree_prior_x_pair_av = u_from_edgelist_df(m_degree_prior_x_pair_av_edgelist_df)

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

# +
saint_prior_wt = get_saint_prior(r_wt)

saint_prior_wt = saint_prior_wt.sort_index(axis=0)
saint_prior_wt = saint_prior_wt.sort_index(axis=1)
r_wt.A_df.sort_index(axis=0, inplace=True)
r_wt.A_df.sort_index(axis=1, inplace=True)

assert np.alltrue(saint_prior_wt.columns == r_wt.A_df.columns)
assert np.alltrue(saint_prior_wt.index == r_wt.A_df.index)
assert np.alltrue(saint_prior_wt.index == r_wt.A_df.columns)

degree_prior_wt = get_degree_prior_from_matrix_df(r_wt.A_df)

pairwise_average_wt = (r_wt.A_df + saint_prior_wt) / 2
# -

degree_prior_x_pair_av_wt_edgelist_df = sa.matrix_df_to_edge_list_df((degree_prior_wt * pairwise_average_wt))
u_degree_prior_x_pair_av_wt = u_from_edgelist_df(degree_prior_x_pair_av_wt_edgelist_df)

predictions = dict(
    saint_prior = u_saint_prior,
    pair_prod = u_pairwise,
    average_edge = r1.u,
    pair_av = u_pairwise_average,
    degree_prior = u_degree_prior,
    deg_x_prod = u_degree_prior_x_pair_prod,
    deg_x_av = u_degree_prior_x_pair_av,
    humap2_hc = gbf.get_humap_high_reference(),
    saint_max = gbf.get_cullin_saint_scores_edgelist(),
    humap2_med = gbf.get_humap_medium_reference(),
    m3 = u_m3,
)

predictions['humap2_hc']._edge_dict

dist.Normal

references = dict(
  costructure = gbf.get_pdb_ppi_predict_cocomplex_reference(),
  direct = gbf.get_pdb_ppi_predict_direct_reference(),
  huri = gbf.get_huri_reference(),
)

references = references | {"indirect" : references['costructure'].edge_identity_difference(
    references['direct'])}

references = references | {
    "decoy149" : gbf.get_decoys_from_u(jax.random.PRNGKey(303), predictions['average_edge'], 149)}

references = references | {
    "decoy23" : gbf.get_decoys_from_u(jax.random.PRNGKey(404), predictions['average_edge'], 23)}

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
    ("saint_max", "costructure"),
    #("humap2_hc", "costructure"),
    #("humap2_med", "costructure"),
    #("m3", "costructure",)
    ),
    multi_save_suffix="_co_structure")

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
    ("saint_max", "direct",)
    #("humap2_hc", "direct"),
    #("humap2_med", "direct")
    #("m3", "direct"),
    ),
    multi_save_suffix="_direct")

ref_key = "indirect"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

ref_key = "huri"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

ref_key = "decoy149"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

ref_key = "decoy23"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

# +
#deg_x_av_top_edge_set
# -

corr_res = gbf.do_benchmark(u_m_corr)

plt.plot(corr_res.ppr_points, corr_res.tpr_points)
plt.show()

corr_res.auc

m_erlp1u20k.hmc_warmup.adapt_state.step_size

# +
# Let's say we want to predict a network based on the ROC type curve
# 1% of the total interaction space
# -

pp_top_node_set, pp_top_edge_set = dcc.get_top_edges_upto_threshold(pairwise_product_edgelist_df, threshold=0.9997)

deg_x_av_top_node_set, deg_x_av_top_edge_set = dcc.get_top_edges_upto_threshold(
    m_degree_prior_x_pair_av_edgelist_df, threshold=0.9)


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

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

net_deg_x_av.show("deg_x_av.html")

deg_x_av_wt_top_node_set, deg_x_av_wt_top_edge_set = dcc.get_top_edges_upto_threshold(
    degree_prior_x_pair_av_wt_edgelist_df, threshold=0.85)

net_dev_x_av_wt = dcc.pyvis_plot_network(deg_x_av_wt_top_edge_set)

net_dev_x_av_wt.show("deg_x_av_wt.html")

x = degree_prior_x_pair_av_wt_edgelist_df
sel = x['a'] == 'vifprotein'
x.sort_values("w", ascending=False).loc[sel, :]

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
