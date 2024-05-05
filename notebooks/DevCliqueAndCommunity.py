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
import numpy as np
from pathlib import Path
import math
import generate_benchmark_figures as gbf
import timeit
import networkx as nx
import _model_variations as mv
import pickle as pkl
from collections import namedtuple


# +
# Load in the average network
def pkl_load(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def av_edge_load(path):
    return pd.read_csv(path, sep="\t")

def get_clique_list(G):
    return list(nx.find_cliques(G))

def get_n_from_iterator(iterator):
    """Memory efficient"""
    n = 0
    for i in iterator:
        n += 1
    return n

def clique_plot(ts, ns):
    plt.plot(ts, ns, 'k.')
    plt.xlabel("threshold")
    plt.ylabel("N cliques")

def get_filter_network(G, threshold):
    filtered_graph = nx.Graph()
    nodes = G.nodes()
    filtered_graph.add_nodes_from(nodes)
    filtered_edges = [(u, v, w) for u,v,w in G.edges(data='weight') if w > threshold]
    filtered_graph.add_weighted_edges_from(filtered_edges)
    return filtered_graph

def add_names_to_df(df, uid2name):
    df.loc[:, "aname"] = [uid2name[uid] for uid in df['auid']]
    df.loc[:, "bname"] = [uid2name[uid] for uid in df['buid']]
    
def get_results(dir_str):
    path = dir_str + "average_predicted_edge_scores.tsv"
    df = av_edge_load(path)
    spec_counts = pd.read_csv(
        dir_str + "spec_table.tsv",
        sep="\t",
        index_col=0)
    return df, spec_counts

def plot_edges(ts, ne):
    plt.plot(ts, ne, 'k')
    plt.ylabel("number of edges")
    plt.xlabel("threshold")
    
def get_n_maximal_cliques(G, ts):
    return [get_n_from_iterator(nx.find_cliques(get_filter_network(G, t))) for t in ts]
    
def h(x):
    return '{:,}'.format(x)


def get_nedges(G, ts):
    return [get_filter_network(G, t).number_of_edges() for t in ts]

def get_graph_from_df(df):
    G = nx.Graph()

    # Add edges with weights to the graph
    for i, r in df.iterrows():
        G.add_edge(r['a_gene'], r['b_gene'], weight=r['w'])
    return G

def CSN_selector(df):
    sel = np.array(["CSN" in r['a_gene'] for i,r in df.iterrows()])
    sel2 = np.array(["CSN" in r['b_gene'] for i,r in df.iterrows()])
    sel3 = sel & sel2
    return sel3

def dataframe_from_matrix(matrix, node_names, name2uid):
    matrix_predictor = []
    for i in range(236):
        for j in range(0, i):
            val = float(matrix[i, j])
            aname = node_names[i]
            bname = node_names[j]
    
            matrix_predictor.append((aname, bname, val))
    
    matrix_predictor_df = pd.DataFrame(
                matrix_predictor, 
                columns=['aname', 'bname', 'w'])

    auid = [name2uid[name] for name in matrix_predictor_df['aname']]
    buid = [name2uid[name] for name in matrix_predictor_df['bname']]
    matrix_predictor_df.loc[:, 'auid'] = auid
    matrix_predictor_df.loc[:, 'buid'] = buid
    return matrix_predictor_df

def add_names_to_df(x, name2uid):
    aname = [name2uid[name] for name in x['auid']]
    bname = [name2uid[name] for name in x['buid']]
    x.loc[:, 'aname'] = aname
    x.loc[:, 'bname'] = bname
    

def plot_benchmark(a, b, c):
    plt.plot(a.ppr_points, a.tpr_points, label="cocomplex")
    plt.plot(b.ppr_points, b.tpr_points, label="direct")
    plt.plot(c.ppr_points, c.tpr_points, label="indirect")
    plt.legend()
    
def get_comparisons(u, dref, iref, cref):
    direct = gbf.do_benchmark(u, dref)
    indirect = gbf.do_benchmark(u, iref)
    cocomplex = gbf.do_benchmark(u, cref)
    return direct, indirect, cocomplex


# +
# Globals
u20_base_path = Path("../results/se_sr_low_prior_1_uniform_all_20k/")

name2uid = gbf.get_cullin_reindexer()
uid2name = {val: key for key, val in name2uid.items()}

pdb_direct = gbf.get_pdb_ppi_predict_direct_reference()
pdb_cocomplex = gbf.get_pdb_ppi_predict_cocomplex_reference()
pdb_indirect = pdb_cocomplex.edge_identity_difference(pdb_direct)


# Paths
BasePaths = namedtuple("BasePaths",
        "wt", "vif", "u20")

base_paths = BasePaths(
    wt = Path("../results/se_sr_wt_ctrl_20k/"),
    vif = Path("../results/se_sr_vif_ctrl_20k/"),)

wt_path = "../results/se_sr_wt_ctrl_20k/"
vif_path = "../results/se_sr_vif_ctrl_20k/"
all_u_20 = "../results/se_sr_low_prior_1_uniform_all_20k/"



u20_mcmc = pkl_load(u20_base_path / "0_model23_se_sr_13.pkl")
u20_model_data = pkl_load(
    u20_base_path / "0_model23_se_sr_13_model_data.pkl")

u20_samples = u20_mcmc['samples']
u20_As = mv.Z2A(u20_samples['z'])
u20_As_av = np.mean(u20_As, axis=0)
u20_A = mv.flat2matrix(u20_As_av, 236)

u_u20 = gbf.model23_matrix2u(u20_A, u20_model_data)

u_u20_control = gbf.model23_results2edge_list(
    Path("../results/se_sr_low_prior_1_uniform_all_20k/"),
    "0_model23_se_sr_13")

# Frames and counts
av_wt, wt_spec_counts = get_results(wt_path)
av_u20, av_u20_spec_counts = get_results(all_u_20)

# Graphs of data frames
Gu20 = get_graph_from_df(av_u20)
Gwt = get_graph_from_df(av_wt)
Gwt_02 = get_filter_network(Gwt, 0.2)

# -

plt.hist(u20_As_av, bins=100)
plt.show()

plot_benchmark(*get_comparisons(u_u20, pdb_direct, pdb_indirect, pdb_cocomplex))

plot_benchmark(*get_comparisons(u_u20_control, pdb_direct, pdb_indirect, pdb_cocomplex))

print(Gu20)
print(Gwt)

# +

print(Gwt_02)
# -

pos = nx.spring_layout(Gwt_02)
nx.draw(Gwt_02, with_labels="True")

# Run clique detection
cliques = list(nx.find_cliques(Gwt_02))

ts = np.linspace(0, 1, 100)
#ns_wt = get_n_maximal_cliques(Gwt, ts)
ne_wt = get_nedges(Gwt, ts)
ne_u20 = get_nedges(Gu20, ts)

plt.plot(ne_wt, label='wt')
plt.plot(ne_u20, label='u20')
plt.legend()

av_wt.loc[CSN_selector(av_wt), :]

clique_plot(ts, ns)

clique_plot(ts, np.log10(ns))
plt.ylabel("log10 N cliques")

ts2 = np.linspace(0.95, 1, 20)
ns2 = [get_n_from_iterator(nx.find_cliques(get_filter_network(G, t))) for t in ts2]

ne2 = [get_filter_network(G, t).number_of_edges() for t in ts2]

plot_edges(ts2, ne2)

clique_plot(ts2, ns2)

ts3 = np.linspace(0.999, 1.0, 20)
ns3 = [get_n_from_iterator(nx.find_cliques(get_filter_network(G, t))) for t in ts3]

ne3 = [get_filter_network(G, t).number_of_edges() for t in ts3]

plot_edges(ts3, ne3)

len(list(nx.enumerate_all_cliques(get_filter_network(G, 0.9998)))[::-1])

# +
# ns3 = [get_n_from_iterator(nx.enumerate_all_cliques(get_filter_network(G, t))) for t in ts3]
# -

a = ['T22D1', 'APBP2', 'CTR9', 'PACS1']
b = ['T22D1', 'HMDH', 'CUL3', 'PAF1']

clique_plot(ts3, ns3)

list(nx.enumerate_all_cliques(get_filter_network(G, 0.99989474)))

list(nx.find_cliques(get_filter_network(G, 0.99973684)))

# +
labels, degree = zip(*list(G2.degree()))

plt.bar(x = np.arange(len(degree)), height=degree)
plt.xticks(np.arange(),b labels=labels)
# -

# Print the edges with their weights
print("Edges with weights:")
for u, v, weight in G.edges.data('weight'):
    print(f"Edge: {u} -- {v}, Weight: {weight}")

ts = np.linspace(0, 1, 100)
neGu = [get_filter_network(Gu20, t).number_of_edges() for t in ts]
neG = [get_filter_network(G, t).number_of_edges() for t in ts]

plt.plot(ts, neGu, label="low uniform")
plt.plot(ts, neG, label="WT")
plt.legend()
plt.xlabel("threshold")
plt.ylabel("Number of edges")
plt.hlines(0.35 * (math.comb(236, 2)), 0, 1, 'r')
plt.hlines(0.2 * (math.comb(236, 2)), 0, 1, 'k')

av_u20

Gu20_filter8 = get_filter_network(Gu20, 0.8)


# +
maximal_cliques = get_clique_list(Gu20_filter8)



clique_len = np.array([len(k) for k in maximal_cliques])
# -

plt.hist(clique_len, bins=100)
plt.show()

Gtemp = get_filter_network(Gu20, 0.99)
maximal_cliques_temp = get_clique_list(Gtemp)
clique_len_temp = np.array([len(k) for k in maximal_cliques_temp])
plt.hist(clique_len_temp, bins=100)
plt.show()

sorted(maximal_cliques_temp, key=(lambda x: len(x)), reverse=True)

sns.heatmap(av_u20_spec_counts.loc[['NRBP2',
  'T22D2',
  'T22D1',
  'ZY11B',
  'KLD10',
  'LEO1',
  'CTR9',
  'LRC41',
  'KLDC3',
  'ZC4H2',
  'HIF1A',
  'UBC12',
  'FEM1B',
  'MED8'], :], xticklabels=4)

plt.matshow(u20_A, cmap='plasma')

sns.clustermap(u20_A)

test = gbf.model23_matrix2u(u20_A)

weighted_degree = jnp.sum(u20_A, axis=0).reshape((236, 1))

weighted_degree_matrix = (weighted_degree @ weighted_degree.T) / np.sum(weighted_degree)

plt.matshow(weighted_degree_matrix, cmap='plasma')

# +
# Prior probability density of an interaction given the degree
degree_prior = weighted_degree / 236

one_minus_degree_prior = 1 - degree_prior
# -

plt.hist(np.ravel(one_minus_degree_prior), bins=100)
plt.show()

mymatshow(one_minus_degree_prior * one_minus_degree_prior.T)

u20_A_corrected = u20_A * (one_minus_degree_prior * one_minus_degree_prior.T)

_N = 12
def mymatshow(x, ax=None):
    plt.matshow(x, cmap='plasma')
    plt.colorbar(shrink=0.8)
plt.figure(figsize=(_N, _N))
mymatshow(u20_A_corrected)

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(u20_A_corrected)

u20_A_corrected_df = pd.DataFrame(
    u20_A_corrected,
    columns = [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])],
    index =   [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])])

u20_corrected_cluster_grid = sns.clustermap(u20_A_corrected_df, metric='correlation')


def matrix_df_to_edge_list_df(df):

    a, b = df.shape
    columns = df.columns
    an = []
    bn = []
    w = []
    for i in range(a):
        for j in range(0, i):
            an.append(columns[i])
            bn.append(columns[j])
            w.append(float(df.iloc[i, j]))
    return pd.DataFrame({'a': an, 'b': bn, 'w': w})


u20_A_corrected_edgelist_df = matrix_df_to_edge_list_df(u20_A_corrected_df)

top_N = 300
k = 0
top_edges = {}
top_node_set = []
for i, r in u20_A_corrected_edgelist_df.sort_values('w', ascending=False).iterrows():
    print(r['a'], r['b'], round(r['w'], 2))
    top_node_set.append(r['a'])
    top_node_set.append(r['b'])
    top_edges[k] = r['a'], r['b']
    if r['w'] < 0.5:
        break
    
    k += 1
top_node_set = list(set(top_node_set))

sns.clustermap(u20_A_corrected_df.loc[top_node_set, top_node_set])

len(top_node_set)

sns.heatmap(av_u20_spec_counts.loc[top_node_set, :], vmax=10)

fig, ax = plt.subplots(1, 2)
ax[0].hist(u20_A_corrected_edgelist_df['w'], bins=100)
ax[1].hist(u20_A_corrected_edgelist_df['w'], bins=100, range=(0.3, 1))
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# +
G = nx.Graph()
G.add_edges_from([(val[0], val[1]) for i, val in top_edges.items()])

net = pyvis.network.Network(notebook = True)

net.from_nx(G)

net.show("example.html")
# -

cbfb_network = ["BAF", "EMD"]
ezh2_network = ["EZH2", "ABCF2"]
nfkb_network = ['NFKB1', "BAG6", "THMS1", "ARHG1", "PDS5A", "RD23B", "OXSR1", "2AAB", "ACAP1", "MOCS1", "KEAP1"]
vif_network = ['vifprotein', 'DCA11', "CUL4B", "SPAG5", "PDC6I", "MAGD1", "2AAB"]
sns.heatmap(av_u20_spec_counts.loc[cbfb_network, :], vmax=10)



# ## CBFB-Beta Network
# Hypothesis - this interaction network occurs at the nuclear envelope.
# - CBFB-RUNX1: Forms the heterodimeric complex core-binding factor (CBF) with CBFB
# - CBFB-RUNX2: Transcription factor
# - CBFB-ANXA1: Plays important roles in the innate immune response as effector of glucocorticoid-mediated responses and regulator of the inflammatory process.
# - CBFB-BAG3: Molecular co-chaperone for HSP70
# - CBFB-AKP8L: Could play a role in constitutive transport element (CTE)-mediated gene expression by association with DHX9. In case of HIV-1 infection, involved in the DHX9-promoted annealing of host tRNA(Lys3) to viral genomic RNA.
# - CBFB-BAF:Non-specific DNA-binding protein. EMD and BAF are cooperative cofactors of HIV-1 infection
# - EMD-BAF!: EMD and BAF are cooperative cofactors of HIV-1 infection. Association of EMD with the viral DNA requires the presence of BAF and viral integrase. The association of viral DNA with chromatin requires the presence of BAF and EMD
# - TOIP1: Required for nuclear membrane integrity. Induces TOR1A and TOR1B
#
# - HPBP1: 
# - CBFB-RF
#
#
#
# ## CUL5 Network
# - CUL5 and substrate receptors ASB1, ASB3, ASB7, ASB13
# - CUL5 and NEDD8
# - CUL5-RB40C: Substrate recognition component cullin
# - CUL5-PCMD2: Elongin BC-CUL5-SOCS-box protein
# - CUL5-RBX2: Probable component of the SCF (SKP1-CUL1-F-box protein) E3 ubiquitin ligase complex 
# - CUL5-EMC1: Endoplasmic reticulmn. Unknown
# - CUL5-UGGG1: ER Quality control
# - CUL5-SOCS4: SCF-like ECS (Elongin BC-CUL2/5-SOCS-box protein) E3 ubiquitin-protein ligase complex
# - CUL5-MYL4: Myosin
# - CUL5-DCNL1: Part of an E3 ubiquitin ligase complex for neddylation. Acts by binding to cullin-RBX1 complexes
# - CUL5-CAND1: Key assembly factor of SCF (SKP1-CUL1-F-box protein) E3 ubiquitin ligase complex
# - CUL5-SYNE2: Linking network between organelles and the actin cytoskeleton to maintain the subcellular spatial organization. As a component of the LINC
#
# ## EZH2 Network
# - EZH2: Polycomb group (PcG) protein. Catalytic subunit of the PRC2/EED-EZH2 complex.
# - ABCF2: ATP-binding cassette sub-family F member 2
#
#
# ## NFKB Network
# - NFKB1: NF-kappa-B is a pleiotropic transcription factor present in almost all cell types.
# - NFKB1-BAG6: (Unkown) Client proteins that cannot be properly delivered to the endoplasmic reticulum are ubiquitinated by RNF126, an E3 ubiquitin-protein ligase associated with BAG6 and are sorted to the proteasome.
# - RD23B: Multiubiquitin chain receptor involved in modulation of proteasomal degradation. Binds to polyubiquitin chains
# - OXSR1: Effector serine/threonine-protein kinase component of the WNK-SPAK/OSR1 kinase cascade
# - Plays a central role in late thymocyte development by controlling both positive and negative T-cell selection.
# - ARGH1: ARHGEF1. 
# - PDS5A: Probable regulator of sister chromatid cohesion in mitosis which may stabilize cohesin complex association with chromatin
# - 2AAB: Interacts with RAD21 a APOB promoter.
#
#
# ## VIF Network
# - Vif-DCA11
# - Vif-CUL4B: Core component of multiple cullin-RING-based E3 ubiquitin-protein ligase complexes. DCA11 known to interact with CUL4A. CUL4B?
# - DCA11-PDC6I: Multifunctional protein involved in endocytosis, multivesicular body biogenesis. Role in HIV-1.
# - CUL4B-SPAG5.Essential component of the mitotic spindle required for normal chromosome segregation and progression into anaphase.
# - CUL4B-MAGD1: Involved in the apoptotic response after nerve growth factor (NGF) binding in neuronal cells. Inhibits cell cycle progression, and facilitates NGFR-mediated apoptosis. May act as a regulator of the function of DLX family members. May enhance ubiquitin ligase activity of RING-type zinc finger-containing E3 ubiquitin-protein ligases
#
# ## LLR1 Network
# - LLR1-SGT1. Substrate recognition subunit of an ECS (Elongin BC-CUL2/5-SOCS-box protein)
# - SGT1. May play a role in ubiquitination and subsequent proteasomal degradation of target proteins.
#
#
#
# ## ARI1 Network
# - ARI1: E3 ubiquitin-protein ligase, which catalyzes ubiquitination of target proteins together with ubiquitin-conjugating enzyme E2 UBE2L3.
# - TFAP4: Transcription factor that activates both viral and cellular genes by binding to the symmetrical DNA sequence 5'-CAGCTG-3'.
#
# ## HSP76 Network
# - HSP76-RNF114: E3 ubiquitin-protein ligase that promotes the ubiquitination of various substrates
# - HSP76-NLRC3: Negative regulator of the innate immune response. HSV-1, TOLL, STING
# - HSP76-STK39: Unkown
#
#
#
#
# RAB21: Small GTPase involved in membrane trafficking control
# - RAB21-CISY: Citrate synthase is found in nearly all cells capable of oxidative metabolism
#
#
#
#
# - RBX2-Nedd8: Part of an E3 ubiquitin ligase complex for neddylation
# - 
#
#
# - APC11 a cullin anaphase protomting complex
#
#
#

import pyvis

cluster_maps = cluster_map_from_clustergrid(
    u20_A_corrected_df,
    u20_corrected_cluster_grid,
    n_clusters=200,
    criterion="max")

for cluster_id, cluster in cluster_maps['col'].items():
    if len(cluster) > 1:
        print(cluster_id, len(cluster))



temp = cluster_maps['col'][94]

for i in temp:
    print(i)

print(temp)
sns.heatmap(u20_A_corrected_df.loc[temp, temp], cmap='plasma', vmin=0, vmax=1)

top_node_set

sns.heatmap(av_u20_spec_counts.loc[, :], vmax=10)

# +
import seaborn as sns
import pandas as pd

# Sample data
data = pd.DataFrame({
    "A": [1, 5, 3, 4, 2],
    "B": [9, 4, 2, 4, 5],
    "C": [6, 8, 7, 5, 6]
}, index = ['D', 'E', 'F', 'G', 'H'])

# Create a clustermap
clustergrid = sns.clustermap(data, method='ward', metric='euclidean', standard_scale=1)


# +
# Row and column reordering
row_order = clustergrid.dendrogram_row.reordered_ind
col_order = clustergrid.dendrogram_col.reordered_ind

print("Row order:", row_order)
print("Column order:", col_order)

# -

cluster_map_from_clustergrid(clustergrid, 3, "max")


def cluster_map_from_clustergrid(data, clustergrid, n_clusters, criterion):
    row_clusters, col_clusters = get_cluster_assignments(clustergrid, n_clusters, criterion)
    row_order = clustergrid.dendrogram_row.reordered_ind
    col_order = clustergrid.dendrogram_col.reordered_ind
    return remap_to_cluster_maps(data, row_clusters, col_clusters, row_order, col_order)


# +
def get_cluster_assignments(clustergrid, n_clusters, criterion="maxclust"):
    # Assuming you want to cut by a specific number of clusters, for example, 2 clusters
    row_clusters = fcluster(clustergrid.dendrogram_row.linkage, n_clusters, criterion='maxclust')
    col_clusters = fcluster(clustergrid.dendrogram_col.linkage, n_clusters, criterion='maxclust')
    return row_clusters, col_clusters

row_clusters, col_clusters = get_cluster_assignments(clustergrid, 3)
print("Row clusters:", row_clusters)
print("Column clusters:", col_clusters)


# +
def remap_to_cluster_maps(data, row_clusters, col_clusters, row_order, col_order):
    # Create a dictionary mapping from cluster number to rows/columns
    row_cluster_map = {i + 1: [] for i in range(row_clusters.max())}
    for idx, cluster_id in enumerate(row_clusters):
        row_cluster_map[cluster_id].append(data.index[row_order[idx]])

    col_cluster_map = {i + 1: [] for i in range(col_clusters.max())}
    for idx, cluster_id in enumerate(col_clusters):
        col_cluster_map[cluster_id].append(data.columns[col_order[idx]])
    return {"row" : row_cluster_map, "col": col_cluster_map}

cluster_maps = remap_to_cluster_maps(row_clusters, col_clusters, row_order, col_order)

print("Row cluster map:", cluster_maps['row'])
print("Column cluster map:", cluster_maps['col'])


# +
fig, ax = plt.subplots(figsize=(10, 5))
dendro_gram_results = sp.cluster.hierarchy.dendrogram(
    cluster_grid.dendrogram_col.linkage,
    p = 5,
    ax=ax,
    truncate_mode='level')

ax.axhline()
plt.show()
# -

dendro_gram_results.keys()

cluster_grid.dendrogram_col.linkage[234, 3]

temp = [234, 202, 136, 20, 199, 53, 30, 20, 45, 155, 12, 197, 8, 97, 59, 21,
       122, 145, 125, 133]
[u20_model_data['node_idx2name'][k] for k in temp]

[u20_model_data['node_idx2name'][k] for k in cluster_grid.data2d.iloc[:, -34:-25].columns]

[u20_model_data['node_idx2name'][k] for k in cluster_grid.data2d.iloc[:, -80:-49].columns]

cluster_grid.data2d.iloc[:, -80:-49].columns

Z_u20 = cluster_grid.dendrogram_col.linkage

clusters = sp.cluster.hierarchy.fcluster(Z_u20, 10, criterion='maxclust' )

clusters

help(sp.cluster.hierarchy.fcluster)

sns.clustermap(u20_A)

plt.hist(u20_A_corrected[np.tril_indices(236, k=-1)], bins=100, range=(0.02, 1))
plt.show()

null_map = u20_A * degree_prior * degree_prior.T

degree = np.sum(u20_A, axis=0).reshape(236, 1)

mymatshow(degree + degree.T)

mymatshow(null_map * 10_000)

mymatshow(u20_A)

plt.matshow(weighted_degree + weighted_degree.T)

plt.hist(weighted_degree_matrix[np.tril_indices(236, k=-1)], bins=100)
plt.show()

plt.matshow(u20_A, cmap="plasma")

# mean of the posterior $E[p(M|D, I)]$
#
#
#

# Prior probability based on seeing the degree
prior_probability_from_degree = weighted_degree_matrix / np.sum(weighted_degree_matrix)


plt.matshow(prior_probability_from_degree)

# Prior probability of seeing
one_minus_prior_probability_from_degree = 1 - prior_probability_from_degree

plt.matshow(one_minus_prior_probability_from_degree, cmap="plasma")
plt.colorbar(shrink=0.8)

u20_post = u20_A * one_minus_prior_probability_from_degree

plt.matshow(u20_post, cmap='plasma', vmin=0, vmax=1)

plt.matshow(u20_A, cmap='plasma')

plt.hist(u20_post[np.tril_indices(236, k=-1)], bins=100)
plt.show()

plt.matshow(prior_probability_from_degree)

plot_benchmark(*get_comparisons(
    u_u20, pdb_direct, pdb_indirect, pdb_cocomplex))

u_u20_A_corrected = gbf.model23_matrix2u(
    u20_A_corrected, u20_model_data)
plot_benchmark(*get_comparisons(
    u_u20_A_corrected, pdb_direct, pdb_indirect, pdb_cocomplex))

x = u_u20_A_corrected
u_u20_A_corrected_df = pd.DataFrame({'auid': x.a_nodes, 'buid': x.b_nodes, 'w': x.edge_values})
add_names_to_df(u_u20_A_corrected_df, uid2name)

u_u20_A_corrected_df.sort_values('w', ascending=False).iloc[0: 50]

help(sns.clustermap)

u20_A_minus_weighted_degree_norm = u20_A - (weighted_degree_matrix / np.max(weighted_degree_matrix))

plt.matshow(u20_A_minus_weighted_degree_norm, cmap="plasma")
plt.colorbar(shrink=0.8)


def min_max_scale_to_01(a):
    return (a - np.min(a)) / (np.max(a)-np.min(a))


u20_scaled_transformed = min_max_scale_to_01(u20_A_minus_weighted_degree_norm)

plt.matshow(u20_scaled_transformed, cmap='plasma', vmin=0, vmax=1)
plt.colorbar(shrink=0.8)

plt.hist(
    min_max_scale_to_01(u20_A_minus_weighted_degree_norm[np.tril_indices(236, k=-1)]),
    bins=100)
plt.show()

# +
# Min max scale
u_u20_scaled_transformed = gbf.model23_matrix2u(
    u20_scaled_transformed, u20_model_data)

plot_benchmark(*get_comparisons(
    u_u20_scaled_transformed,
    pdb_direct, pdb_indirect, pdb_cocomplex))
# -

# ?gbf.model23_matrix2u

matrix_ratio = u20_A / weighted_degree_matrix
AD_HOC_NORM_CONST = 78.5194
matrix_ratio = matrix_ratio * AD_HOC_NORM_CONST

plt.matshow(matrix_ratio, cmap='plasma')
plt.colorbar(shrink=0.8)

plt.hist(matrix_ratio[np.tril_indices_from(matrix_ratio, k=-1)], bins=100)
plt.show()

plt.hist(matrix_ratio[np.tril_indices_from(matrix_ratio, k=-1)], bins=100, range=(0.1, 1))
plt.show()

node_names = [u20_model_data['node_idx2name'][i] for i in range(len(matrix_ratio))]
matrix_ratio_df = pd.DataFrame(matrix_ratio,
                              columns=node_names,
                              index=node_names)

indices = np.where(matrix_ratio_df > 0.4)

matrix_ratio_df.iloc[indices[0], indices[1]]

# +
reindexer = gbf.get_cullin_reindexer()

matrix_ratio_predictor_df = dataframe_from_matrix(matrix_ratio,
                                                 node_names, name2uid)
# -



u = gbf.UndirectedEdgeList()
u.update_from_df(matrix_ratio_predictor_df, 
                 a_colname='auid',
                 b_colname='buid',
                 edge_value_colname='w',
                 multi_edge_value_merge_strategy='max')

direct_comparison = gbf.do_benchmark(u, pdb_direct)
indirect_comparison = gbf.do_benchmark(u, pdb_indirect)
cocomplex_comparison = gbf.do_benchmark(u, pdb_cocomplex)

plt.plot(cocomplex_comparison.ppr_points, cocomplex_comparison.tpr_points, label="cocomplex")
plt.plot(direct_comparison.ppr_points, direct_comparison.tpr_points, label="direct")
plt.plot(indirect_comparison.ppr_points, indirect_comparison.tpr_points, label="indirect")

plt.matshow(weighted_degree_matrix, cmap='plasma')

norm_weighted_degree_matrix = weighted_degree_matrix / np.max(weighted_degree_matrix)



plt.matshow(u20_A)

u20_m

u_u20 = gbf.model23_matrix2u(u20_A, u20_model_data)

hash(u_u20)

u20_average_edge.edge_values

u_u20.edge_values

u20_control_direct_comparison = gbf.do_benchmark(u20_average_edge, pdb_direct)
u20_control_indirect_comparison = gbf.do_benchmark(u20_average_edge, pdb_indirect)
u20_control_cocomplex_comparison = gbf.do_benchmark(u20_average_edge, pdb_cocomplex)

u20A_minus_weighted_degree = u20_A - norm_weighted_degree_matrix

plt.matshow(u20A_minus_weighted_degree, cmap='plasma')
plt.colorbar(shrink=0.8)

u20_predictor_df = dataframe_from_matrix(u20_A, node_names, name2uid)

u20_average_edge = gbf.model23_results2edge_list(
    Path("../results/se_sr_low_prior_1_uniform_all_20k/"),
    "0_model23_se_sr_13")

 

u20_average_edge

plot_benchmark(u20_control_cocomplex_comparison,
               u20_control_direct_comparison,
               u20_control_indirect_comparison)

u_u20 = gbf.UndirectedEdgeList()
u_u20.update_from_df(
    u20_predictor_df, 
    a_colname='auid',
    b_colname='buid',
    edge_value_colname='w',
    multi_edge_value_merge_strategy='max')

u20_direct_comparison = gbf.do_benchmark(u_u20, pdb_direct)
u20_indirect_comparison = gbf.do_benchmark(u_u20, pdb_indirect)
u20_cocomplex_comparison = gbf.do_benchmark(u_u20, pdb_cocomplex)

plt.plot(u20_cocomplex_comparison.ppr_points, u20_cocomplex_comparison.tpr_points, label="cocomplex")
plt.plot(u20_direct_comparison.ppr_points, u20_direct_comparison.tpr_points, label="direct")
plt.plot(u20_indirect_comparison.ppr_points, u20_indirect_comparison.tpr_points, label="indirect")

plt.hist(u20A_minus_weighted_degree[np.tril_indices(236, k=-1)], bins=100)
plt.xlabel("Difference")
plt.show()

u20_model_data['node_idx2name']

np.sum(matrix_ratio > 0.25)

plt.hist(matrix_ratio[np.tril_indices_from(matrix_ratio, k=-1)], bins=100)
plt.show()

reciprocal_weighted_degree = 1 / weighted_degree

reciprocal_matmul = reciprocal_weighted_degree @ reciprocal_weighted_degree.T

plt.matshow(reciprocal_matmul)

normalized_weighted_degree = weighted_degree / jnp.sum(weighted_degree)

reciprocal_normalized_weighted_degree = 1 / normalized_weighted_degree

plt.plot(weighted_degree, 'k.')

plt.matshow(weighted_degree.T)

Gu20_filter4 = get_filter_network(Gu20, 0.4)

print(Gu20_filter4)

plot_edges(ts, ne)

Av_u20_matrix

av_u20

Av_u20_matrix[0, 5]

Av_u20_matrix = mv.flat2matrix(av_u20['w'].values, 236)

sns.heatmap(Av_u20_matrix, cmap='plasma')

weighted_degree = jnp.sum(Av_u20_matrix, axis=1)



plt.plot(weighted_degree)
plt.ylabel("Weighted Degree")
plt.xlabel("Node ID")

d
