"""
Implementation for Binary undirected edges read from tables

1. Edges are unique.
2. Edge weights are 1, 0 edges are not represented. 
3. Edges cannot be self edges.
4. If any pair is in the input table the edge is represented

Edge ordering
- a > b
u = EdgeList
"""

import pandas as pd
import pickle as pkl
import numpy as np
from functools import partial

from undirected_edge_list import UndirectedEdgeList
import tpr_ppr

def load_cullin_composite_saint_scores():
    df = pd.read_csv("../data/processed/cullin/composite_table.tsv", sep="\t")
    return df

def load_sars2_composite_saint_scores():
    df = pd.read_csv("../data/processed/gordon_sars_cov_2/composite_table.tsv", sep="\t")
    return df

def get_cullin_saint_scores_edgelist():
    df = load_cullin_composite_saint_scores()
    c = UndirectedEdgeList()
    c.update_from_df(df, a_colname="Bait", b_colname="Prey", edge_value_colname = "MSscore",
                     multi_edge_value_merge_strategy = "max")
    # Reindex
    reindexer = get_cullin_reindexer()
    c.reindex(reindexer)
    return c

def get_sars2_reindexer():
    reindex_df = pd.read_csv("../data/processed/cullin/id_map.tsv", sep="\t", names=["PreyGene", "uid"])
    reindexer = {}
    for i, r in reindex_df.iterrows():
        prey_gene = r['PreyGene']
        prey_uid = r['uid']
        assert prey_gene not in reindexer
        reindexer[prey_gene] = prey_uid 
    return reindexer


def get_sars2_saint_scores_edgelist():
    df = load_sars2_composite_saint_scores() 
    c = UndirectedEdgeList()
    c.update_from_df(df, a_colname="Bait", b_colname="Prey", edge_value_colname = "MSscore",
                     multi_edge_value_merge_strategy = "max")
    # Reindex
    reindexer = get_sars2_reindexer()
    c.reindex(reindexer)
    return c

def get_cullin_reindexer():
    reindex_df = pd.read_csv("../data/processed/cullin/id_map.tsv", sep="\t", names=["PreyGene", "uid"])
    reindexer = {}
    for i, r in reindex_df.iterrows():
        prey_gene = r['PreyGene']
        prey_uid = r['uid']
        assert prey_gene not in reindexer
        reindexer[prey_gene] = prey_uid 
    return reindexer

def get_huri_reference():
    df = pd.read_csv("../data/processed/references/HuRI_reference.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u 

def get_humap_medium_reference():
    df = pd.read_csv("../data/processed/references/humap2_ppis_medium.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u

def get_humap_high_reference():
    df = pd.read_csv("../data/processed/references/humap2_ppis_high.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u

def get_biogrid_reference():
    df = pd.read_csv("../data/processed/references/biogrid_reference.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u

def get_pdb_ppi_predict_direct_reference(source="tsv"):
    if source == "pkl":
        with open("./direct_benchmark.pkl", "rb") as f:
            ref = pkl.load(f)
            direct_matrix = ref.reference.matrix
            df = xarray_matrix2edge_list_df(direct_matrix)
    elif source == "tsv":
        df = pd.read_csv("../data/processed/references/pdb_ppi_prediction/direct_benchmark.tsv")
    else:
        raise NotImplementedError
    u = UndirectedEdgeList()
    u.update_from_df(df)
    reindexer = get_cullin_reindexer()
    u.reindex(reindexer, enforce_coverage = False)
    return u 

def get_pdb_ppi_predict_cocomplex_reference(source="tsv"):
    if source == "pkl":
        with open("./cocomplex_benchmark.pkl", "rb") as f:
            ref = pkl.load(f)
            xar = ref.reference.matrix
            df = xarray_matrix2edge_list_df(xar)
    elif source == "tsv":
        df = pd.read_csv("../data/processed/references/pdb_ppi_prediction/cocomplex_benchmark.tsv")
    else:
        raise NotImplementedError
    u = UndirectedEdgeList()
    u.update_from_df(df)
    reindexer = get_cullin_reindexer()
    u.reindex(reindexer, enforce_coverage = False)
    return u 

def get_intact_all_first_uid_reference():
    df = pd.read_csv("../data/processed/references/intact/intact_all_first_uid.tsv")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u 

def get_all_intersection_datasets():
    return {"intact_all_first_uid" : get_intact_all_first_uid_reference(),
            "biogrid_all" : get_biogrid_reference(),
            "huri" : get_huri_reference(),
            "humap_high" : get_humap_high_reference(),
            "humap_medium" : get_humap_medium_reference(),
            "pdb_ppi_direct" : get_pdb_ppi_predict_direct_reference(),
            "pdb_ppi_cocomplex" : get_pdb_ppi_predict_cocomplex_reference(),
            "cullin_max_saint" : get_cullin_saint_scores_edgelist(),
#            "sars2_max_saint" : get_sars2_saint_scores_edgelist(),
            }


def xarray_matrix2edge_list_df(xar):
    a = []
    b = []
    val = []
    N = xar.shape[0]
    ids = xar.preyu.values
    for i in range(N):
        for j in range(0, i):
            value = xar[i, j].item()
            if value > 0:
                a.append(ids[i])
                b.append(ids[j])
                val.append(value)
    return pd.DataFrame({'auid': a, 'buid': b, 'val': val})

#ds_list = [("huri", h), ("humap_med", humap_medium), ("humap_high", humap_high), ("biogrid_all", biogrid_reference),
#          ("pdb_ppi_direct", direct_ref), ("pdb_ppi_cocomplex", cocomplex_ref), ("intact_all", intact_all), ("cullin_saint", c),
#           ("cullin_all_by_all", c_dense)]



def get_dataset_matrix(ds_list, intersection_method):
    """
    ds_list is a list of tuples each a ("name", u: UndirectedEdgeList)
    method: one of 'edge' or 'node'
    """
    if isinstance(ds_list, dict):
        ds_list = [x for x in ds_list.items()]
    if intersection_method == "edge":
        def calculator(a, b):
            return len(a.edge_identity_intersection(b))
    elif intersection_method == "node":
        def calculator(a, b):
            return len(a.node_intersection(b))
    else:
        raise NotImplementedError
    N = len(ds_list)
    results = np.zeros((N, N), dtype = int)
    for i in range(N):
        for j in range(0, i+1):
            a = ds_list[i][1]
            b = ds_list[j][1]
            results[i, j] = calculator(a, b) 
    names = [t[0] for t in ds_list] 
    df = pd.DataFrame(results, index=names, columns = names)
    return df

def benchmark_cullin_max_saint(ds_dict):
    results = {} 
    for key, e in ds_dict.items():
        x = tpr_ppr.PprTprCalculator(pred = ds_dict['cullin_max_saint'], ref = e)
        r = x.crunch()
        results[key] = (r.auc, r.shuff_auc)
    return results




get_edge_intersection_matrix = partial(get_dataset_matrix, intersection_method = "edge")
get_node_intersection_matrix = partial(get_dataset_matrix, intersection_method = "node")

def main():
    outdir = "../results/generate_benchmark_figures/"
    ...
if __name__ == "__main__":
    main()
