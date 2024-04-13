import click
import pandas as pd
from pathlib import Path
from undirected_edge_list import UndirectedEdgeList
import matplotlib.pyplot as plt
import numpy as np

@click.command()
@click.option("--a")
@click.option("--b")
@click.option("--o")
def main(a, b, o): 
    _main(a, b, o)

def generate_plots(asymetric_difference, o):
    ...

def get_asymetric_difference(a_edges: UndirectedEdgeList, b_edges: UndirectedEdgeList):
    def update_shared_edges(edge_difference, a_edges, b_edges):
        def update_edge_difference_from_only_a_edge_ids(only_a_edge_ids, a_edges, b_edges, edge_difference):
            """
            The difference is equal to a
            """
            for edge in only_a_edge_ids:
                edge_difference[edge] = a_edges._edge_dict[edge]
            return edge_difference
        def update_edge_difference_from_shared_edge_ids(shared_edge_ids, a_edges, b_edges, edge_difference):
            """
            The difference is equal to the difference of edges
            """
            for edge in shared_edge_ids:
                edge_difference[edge] = a_edges._edge_dict[edge] - b_edges._edge_dict[edge]
            return edge_difference
        def update_edge_difference_from_only_b_edge_ids(only_b_edge_ids, a_edges, b_edges, edge_difference):
            """
            The difference is equal to negative b
            """
            for edge in only_b_edge_ids:
                edge_difference[edge] = -b_edges._edge_dict[edge]
            return edge_difference


        a_edge_ids = set(a_edges._edge_dict.keys())
        b_edge_ids = set(b_edges._edge_dict.keys())

        only_a_edge_ids = a_edge_ids - b_edge_ids
        shared_edge_ids = a_edge_ids.intersection(b_edge_ids)
        only_b_edge_ids = b_edge_ids - a_edge_ids

        edge_difference = update_edge_difference_from_only_a_edge_ids(only_a_edge_ids, a_edges, b_edges, edge_difference)
        edge_difference = update_edge_difference_from_shared_edge_ids(shared_edge_ids, a_edges, b_edges, edge_difference)
        edge_difference = update_edge_difference_from_only_b_edge_ids(only_b_edge_ids, a_edges, b_edges, edge_difference)

        return edge_difference

    def undirected_edge_list_from_edge_dict(edge_difference):
        o = UndirectedEdgeList() 
        a_gene = []
        b_gene = []
        diff = []
        for edge, value in edge_difference.items():
            a, b = edge
            a_gene.append(a)
            b_gene.append(b)
            diff.append(value)
        df = pd.DataFrame({"a_gene": a_gene, "b_gene": b_gene, "diff": diff})
        o.update_from_df(df, a_colname = "a_gene", b_colname = "b_gene", edge_value_colname = "diff", multi_edge_value_merge_strategy="max")
        return o
        
    a_edges._build_edge_dict()
    b_edges._build_edge_dict()
    
    edge_difference = {}
    edge_difference = update_shared_edges(edge_difference, a_edges, b_edges)
    return undirected_edge_list_from_edge_dict(edge_difference) 
    
def get_edges(edge_path) -> UndirectedEdgeList:
    u = UndirectedEdgeList() 
    df = pd.read_csv(edge_path, sep="\t")
    u.update_from_df(df,  a_colname="a_gene", b_colname="b_gene", edge_value_colname="w", multi_edge_value_merge_strategy="max")
    return u 

def pair_difference_comparison(a_fpath: Path, b_fpath: Path, o_dirpath: Path):
    """
    Does an assymetric comparison between a and b generating figures and writing to output directory
    """

    a_edges = get_edges(a_fpath)
    b_edges = get_edges(b_fpath)

    diff_name = a_fpath.parent.stem + "__" + b_fpath.parent.stem

    asymetric_difference = get_asymetric_difference(a_edges, b_edges)
    generate_pair_plots_and_tables(diff_name, asymetric_difference, o_dirpath)


def generate_pair_plots_and_tables(diff_name: str, diff: UndirectedEdgeList, o_dirpath: Path):    
    def save(name):
        plt.savefig(str(o_dirpath /  f"{diff_name}_{name}_300.png"), dpi=300)
        plt.savefig(str(o_dirpath /  f"{diff_name}_{name}_1200.png"), dpi=1200)
        plt.close()
    
    def gen_diff_histogram():
        fig, ax = plt.subplots()
        plt.hist(np.array(diff.edge_values), bins=100)
        plt.title(diff_name)
        plt.xlabel("Edge score difference")
        plt.ylabel("Frequency")
        save("hist")
    
    def save_diff_table():
        diff.to_csv(o_dirpath / f"{diff_name}.tsv", a_colname="a_gene", b_colname="b_gene", sep="\t", index=False, header=True,
                    sort_values = True, edge_colname="diff")

    gen_diff_histogram()
    save_diff_table()





def _main(a_dirpath, b_dirpath, o_dirpath): 
    a_dirpath = Path(a_dirpath)
    b_dirpath = Path(b_dirpath)
    o_dirpath = Path(o_dirpath)

    assert a_dirpath.is_dir()
    assert b_dirpath.is_dir()
    if not o_dirpath.is_dir():
        o_dirpath.mkdir()

    a_fpath = a_dirpath / "average_predicted_edge_scores.tsv"
    b_fpath = b_dirpath / "average_predicted_edge_scores.tsv"

    assert a_fpath.is_file()
    assert b_fpath.is_file()

    pair_difference_comparison(a_fpath, b_fpath, o_dirpath)


_examples = {"se_sr_wt_ctrl_20k" : Path("../results/se_sr_wt_ctrl_20k/average_predicted_edge_scores.tsv"),
             "se_sr_all_20k" : Path("../results/se_sr_all_20k/average_predicted_edge_scores.tsv")}

def _example():
    a = get_edges(_examples['se_sr_wt_ctrl_20k'])
    b = get_edges(_examples['se_sr_all_20k'])
    return a, b

if __name__ == "__main__":
    main()
