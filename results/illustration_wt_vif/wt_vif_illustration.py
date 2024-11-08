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
import seaborn as sns
import jax
import pickle as pkl
import tasteful_tools as tt
import math
import numpy as np

import sys
sys.path.append("../../notebooks")
import _model_variations as mv
import generate_sampling_figures as gsf
import generate_benchmark_figures as gbf
import data_io
import sampling_assessment as sa
from collections import defaultdict, namedtuple
from pathlib import Path



def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)
    
def edge_table2edge_matrix(edge_table):
    genes = sorted(list(set(edge_table['a_gene'].values).union(edge_table['b_gene'].values)))
    N = len(genes)
    matrix = np.zeros((N, N))
    df = pd.DataFrame(matrix, index=genes, columns=genes)
    for label, row in edge_table.iterrows():
        a = row["a_gene"]
        b = row["b_gene"]
        value = row["w"]
        df.loc[a, b] = value
        df.loc[b, a] = value
    return df


# -

# Load in the data
def get_average_edge_table(kw="wt"):
    edge_table = None
    matrix = None

    assert_cols = ["auid", "buid", "a_gene", "b_gene"]

    N_av = 0
    for i in range(1, 21):
        f_path = Path(f"../mini_model23_n_{kw}_20k_rseed_1/average_predicted_edge_scores_{i}.tsv")

        if f_path.is_file():
            l_edge_table = pd.read_csv(
                f"../mini_model23_n_{kw}_20k_rseed_1/average_predicted_edge_scores_{i}.tsv", sep="\t")

            l_matrix = edge_table2edge_matrix(l_edge_table)
            if matrix is None:
                matrix = l_matrix
            else:
                assert np.alltrue(l_matrix.columns == matrix.columns)
                N_av +=1
                matrix = matrix + l_matrix
    return matrix / N_av, N_av
            


wt_av_mat, N_av_wt = get_average_edge_table()

vf_av_mat, N_av_vf = get_average_edge_table(kw="vif")

plt.hist(vf_av_mat.values[np.tril_indices(len(vf_av_mat))] / N_av_vf, bins=100)
plt.show()

f"{wt_av_mat.shape} | {vf_av_mat.shape}"

f"{math.comb(177, 2)} | {math.comb(186, 2)}"

f"{186 - 177} protein types present in dVif and not in WT"


def get_unique_ids(edge_table):
    return set(edge_table["a_gene"].values).union(set(edge_table["b_gene"].values))


vf_ids = (set(vf_av_mat.columns) - set(wt_av_mat.columns))
wt_ids = (set(wt_av_mat.columns) - set(vf_av_mat.columns))
shared_ids = set(wt_av_mat.columns).intersection(vf_av_mat.columns)

f"{vf_ids - wt_ids}"

f"{wt_ids - vf_ids}"

# +
vf_only = vf_ids - wt_ids
wt_only = wt_ids - vf_ids
only_in_one_ids = (vf_only).union(wt_only)

vf_only = list(vf_only)
wt_only = list(wt_only)
only_in_one_ids = list(only_in_one_ids)
# -

wt_av_mat_flat = wt_av_mat.values[np.tril_indices(wt_av_mat.shape[0], k=-1)]

# +
ts = []
ps = []
for t in np.linspace(min(wt_av_mat_flat), max(wt_av_mat_flat), 1000):
    ts.append(t)
    ps.append(np.sum(wt_av_mat_flat >= t) / len(wt_av_mat_flat))
    
plt.plot(ts, ps)
plt.xlabel("Threshold")
plt.ylabel("Percent")

# We select the threshold based on the ROC curve
    
# -

min(wt_av_mat_flat)

sns.heatmap(wt_av_mat.loc[wt_only, wt_only], cmap="binary",
           xticklabels=wt_only, yticklabels=wt_only, vmin=0, vmax=0.5,
           cbar_kws = {"label": "Average edge value", "location": "left"})
plt.tight_layout()
plt.savefig("Fig6_300.png", dpi=300)
plt.savefig("Fig6_1200.png", dpi=1200)

wt_spec_table = pd.read_csv("../mini_model23_n_wt_20k_rseed_1/spec_table.tsv", sep="\t", index_col=0)

sns.heatmap(wt_av_mat.loc[compare_to_input_info, compare_to_input_info], cmap="binary", vmin=0, vmax=0.5)

wt_spec_table.loc[compare_to_input_info, :]

compare_to_input_info = ["PDC6I", "DCA11", "CUL4A", "CUL4B",]
sns.heatmap(wt_spec_table.loc[compare_to_input_info, :], cmap = "magma",
           cbar_kws = {"label": "spectral count", "location": "left"})
plt.tight_layout()
#plt.savefig("Fig6b_300.png", dpi=300)
#plt.savefig("Fig6b_1200.png", dpi=300)


# ## Interpretation
#
# DBB1 and CUL4A are required for cell cycle arrest. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2045451/
# DBB1 binds directly to CUL4A. https://pubmed.ncbi.nlm.nih.gov/16964240/
#
# Predicts an interaction between PDC6I and DBB1-CUL4 in a Vif dependant manner.
# PDC6I (ALIX, AIP1, KIAA1375)
#
# Nodes that are present in the wt condition but not in the dVif condition
#
# DCA11 and CUL4A are members of the Cul4-RING E3 ubiquitin ligase complex.  
# CUL4A is 28% sequence identical to CUL5.  
# Both complexes contain the ELOB/ELOC subcomplex.  
#
# Further, PDC6I is predicted to interact both with CUL4A and DCA11 and 

threshold = 0.45
hc_interactions = [label for label, row in wt_av_mat.iterrows() if np.max(row.values) > threshold]

results = sns.clustermap(wt_av_mat.loc[hc_interactions, hc_interactions], cmap="binary",
           xticklabels=hc_interactions, yticklabels=hc_interactions, vmin=0, vmax=0.5)


# +
def heatmap_helper(mat, sel):
    sns.heatmap(mat.loc[sel, sel], cmap="binary",
           xticklabels=sel, yticklabels=sel, vmin=0, vmax=0.5)
    
cluster1 = ["RUNX1", "RUNX2", "APC11", "CAND1"]

# -

cluster2 = ["CSN1", "CSN7B", "SOCS6", "LRC41", "WDR61", 
            "CSN4", "KLDC1", "ASB8", "CSN3", "CSN8", "CSN5", "CSN6", "CSN2", "CSN7A"]

heatmap_helper(wt_av_mat, cluster2)

heatmap_helper(wt_av_mat, cluster1)

sns.heatmap(vf_av_mat.loc[vif_only, vif_only],
           xticklabels = vif_only, yticklabels = vif_only)
plt.tight_layout()

hc_interactions = [label for label, r in wt]

if isinstance(shared_ids, set):
    shared_ids = sorted(shared_ids)
difference_map = wt_av_mat.loc[shared_ids, shared_ids] - vf_av_mat.loc[shared_ids, shared_ids]
Ndiff, _ = difference_map.shape

shared_ids

plt.hist(np.ravel(difference_map.values[np.tril_indices(Ndiff)]), bins=100)
plt.vlines(-0.2, 0, 5000, 'r')
plt.vlines(0.2, 0, 5000, 'r')

heatmap(difference_map > 0.3)

# +
# Top wt edges
diffN, _ = difference_map.shape
cols = difference_map.columns
alist = []
blist = []
vlist = []
for i in range(diffN):
    for j in range(i, diffN):
        value = difference_map.iloc[i, j]
        a = cols[i]
        b = cols[j]
        alist.append(a)
        blist.append(b)
        vlist.append(value)
        
diff_list = pd.DataFrame({"a_gene": alist, "b_gene": blist, "w": vlist})
diff_list.sort_values("w", inplace=True)
        
# -

sns.barplot(diff_list.iloc[0:100])

difference_map

# # Difference map near CUL5

# +
cul5_cols = ["CUL5", "ELOB", "ELOC", "PEBB", "vifprotein"]

def heatmap(x):
    sns.heatmap(x, vmin=-0.4, vmax=0.4)


heatmap(difference_map.loc[cul5_cols, cul5_cols])
# -

heatmap(wt_av_mat.loc[cul5_cols, cul5_cols])

heatmap(vf_av_mat.loc[cul5_cols, cul5_cols])

pebb_cols = ["PEBB", "RUNX1", "RUNX2", "RUNX3"]
heatmap(difference_map.loc[pebb_cols, pebb_cols])

f"N only in one {len(only_in_one_ids)}"

# +
# Differential Nodes
# -

plt.hist(vif_edge_table['w'], bins=100)
plt.show()

# +
# Read in the RF2 PPI-list

rf2_path = "../../data/rf2/RF_scores"
rf2_scores = pd.read_csv(rf2_path, sep="\t", names=["pair", "rf2-score"])

# +
# Get the rf2 scores corresponding to the CRL4 modeled network

# Read in all the conditions (wt, vif, mock, all) and get the set of uniprot ids

# Write these ids to a pattern file

# # copy the pattern file with the preceding _

# use regex to write a new file of PPI scores corresponding to the CRL5 network

# +
uid_set = set()
temp = pd.read_csv(
    "../mini_model23_n_all_20k_merged/average_predicted_edge_scores.tsv", sep="\t")

uid_set = set(temp["auid"]).union(set(temp["buid"]))
print(len(uid_set))
temp = pd.read_csv(
    "../mini_model23_n_wt_20k_rseed_1/average_predicted_edge_scores.tsv", sep="\t")
uid_set = uid_set.union(set(temp["auid"]).union(set(temp["buid"])))
print(len(uid_set))
temp = pd.read_csv(
    "../mini_model23_n_vif_20k_rseed_1/average_predicted_edge_scores.tsv", sep="\t")
uid_set = uid_set.union(set(temp["auid"]).union(set(temp["buid"])))
print(len(uid_set))
temp = pd.DataFrame({"uid" : sorted(list(uid_set))})
temp.to_csv("CRL5_UIDS", index=None, header=None)
# -

temp

all_average_predicted_edge_scores
