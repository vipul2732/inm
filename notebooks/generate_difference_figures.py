import click
import pandas as pd
from pathlib import Path
from undirected_edge_list import UndirectedEdgeList
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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

def pair_difference_comparison(a_fpath: Path, b_fpath: Path, o_dirpath: Path, prefix=""):
    """
    Does an assymetric comparison between a and b generating figures and writing to output directory
    """

    a_edges = get_edges(a_fpath)
    b_edges = get_edges(b_fpath)

    diff_name = prefix + a_fpath.parent.stem + "__" + b_fpath.parent.stem

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

def add_edges(G: nx.Graph, df: pd.DataFrame, name, mag = 0.95):
    for i, r in df.iterrows():
        val = r['diff']
        if abs(val) >= mag:
            G.add_edge(r['a_gene'], r['b_gene'], weight=val, name=name)

def plot_diffmap(df, name, mag = 0.95):
    G = nx.Graph()
    add_edges(G, df, name, mag = mag)
    pos = nx.circular_layout(G)
    # Function to scale label positions
    def scale_label_pos(pos, offset=0.1):
        mean_x = np.mean([p[0] for p in pos.values()])
        mean_y = np.mean([p[1] for p in pos.values()]) + 1
        scaled_pos = {node: [(p[0] - mean_x) * 1.1 + mean_x, (p[1] - mean_y) * 1.1 + mean_y + offset] for node, p in pos.items()}
        return scaled_pos

    def calculate_text_rotation(pos):
        rotations = {}
        center_x = np.mean([p[0] for p in pos.values()])
        center_y = np.mean([p[1] for p in pos.values()])

        for node, (x, y) in pos.items():
            angle = np.arctan2(y - center_y, x - center_x)
            rotation = np.degrees(angle) #+ 90  # offset to align from radial line
            if abs(rotation) > 90:  # adjust rotation to keep text upright
                rotation -= 180
            rotations[node] = rotation
        return rotations

    # Get text rotations
    text_rotations = calculate_text_rotation(pos)

    # Scale positions
    scaled_label_pos = scale_label_pos(pos)
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    ax1.set_box_aspect(1)
    axs = [ax1, ax2]
    #plt.suptitle(name, y=0.82, fontsize=16)
    ax1.set_title(name, fontsize=16)
    #ax1 = fig.add_axes([0, 0, 0.8, 1.])
    #ax2 = fig.add_axes([0.8, 0, 0.2, 1.])
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=axs[0], node_size=40, node_color='skyblue', alpha=0.8)
    
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    norm = plt.Normalize(min(weights), max(weights))
    color_scale=1
    norm = plt.Normalize(-color_scale, color_scale)
    cmap = plt.cm.coolwarm
    cmap = plt.cm.cividis
    cmap = plt.cm.plasma
    cmap = plt.cm.PuOr
    cmap = plt.cm.seismic
    edge_colors = [cmap(norm(weight)) for weight in weights]
    
    nx.draw_networkx_edges(G, pos, ax=axs[0], edge_color=edge_colors)

    # Draw scaled labels
    #nx.draw_networkx_labels(G, scaled_label_pos, font_size=12, font_color='black')

    # Draw labels with specific rotations
    for node, (x, y) in scaled_label_pos.items():
        axs[0].text(x, y, node, rotation=text_rotations[node], horizontalalignment='center', verticalalignment='center', fontdict={'color': 'black', 'size': 12})
    
    # Plot color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #ax = plt.gca()
    cbar = plt.colorbar(sm, ax=axs[1], shrink=0.8, location="left", pad=0, fraction=0.4, ticks=[])
    if "-" in name: 
        a, b = name.split("-")
    elif "__" in name:
        a, b = name.split("__")
    else:
        raise ValueError(f"Expected name seperator")
    a = a + " only"
    b = b + " only"
    # Adding labels to the top and bottom of the colorbar
    cbar.ax.text(1.0, 1.01, a, transform=cbar.ax.transAxes, ha='right', va='center')
    cbar.ax.text(1.0, -0.01, b, transform=cbar.ax.transAxes, ha='right', va='center')


    # Show plot
    
    plt.axis('off')
    #plt.show()
    #plt.close()
    return sm

def plot_diffmap_at_multiple_thresholds(diff_df, diff_name, o_dirpath):
    diff_name = diff_name.removesuffix(".tsv")
    thresholds = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.8]
    for t in thresholds:
        plot_diffmap(diff_df, name = diff_name, mag=t) 
        str_t = str(t)
        str_t = str_t.removeprefix("0.")
        basename = f"{diff_name}_{str_t}"
        plt.savefig(o_dirpath / (basename + "_300.png"), dpi=300)
        plt.savefig(o_dirpath / (basename + "_1200.png"), dpi=300)
        plt.close()


def do_compare(a_fpath, b_fpath, o_dirpath, prefix=""):
    assert a_fpath.is_file()
    assert b_fpath.is_file()
    pair_difference_comparison(a_fpath, b_fpath, o_dirpath, prefix=prefix)
    diff_name = prefix + a_fpath.parent.stem + "__" + b_fpath.parent.stem + ".tsv"
    diff_df = pd.read_csv(o_dirpath / diff_name, sep="\t")
    plot_diffmap_at_multiple_thresholds(diff_df, diff_name, o_dirpath)
    

def _main(a_dirpath, b_dirpath, o_dirpath): 
    a_dirpath = Path(a_dirpath)
    b_dirpath = Path(b_dirpath)
    o_dirpath = Path(o_dirpath)

    assert a_dirpath.is_dir()
    assert b_dirpath.is_dir()
    if not o_dirpath.is_dir():
        o_dirpath.mkdir()

    a_fpath = a_dirpath / "average_predicted_edge_scores_filter3.tsv"
    b_fpath = b_dirpath / "average_predicted_edge_scores_filter3.tsv"

    do_compare(a_fpath, b_fpath, o_dirpath, prefix="filter3_")

    a_fpath = a_dirpath / "average_predicted_edge_scores.tsv"
    b_fpath = b_dirpath / "average_predicted_edge_scores.tsv"

    do_compare(a_fpath, b_fpath, o_dirpath)

    a_fpath = a_dirpath / "average_predicted_edge_scores_filter10.tsv"
    b_fpath = b_dirpath / "average_predicted_edge_scores_filter10.tsv"

    do_compare(a_fpath, b_fpath, o_dirpath, prefix="filter10_")


_examples = {"se_sr_wt_ctrl_20k" : Path("../results/se_sr_wt_ctrl_20k/average_predicted_edge_scores.tsv"),
             "se_sr_all_20k" : Path("../results/se_sr_all_20k/average_predicted_edge_scores.tsv")}

def _example():
    a = get_edges(_examples['se_sr_wt_ctrl_20k'])
    b = get_edges(_examples['se_sr_all_20k'])
    return a, b

if __name__ == "__main__":
    main()
