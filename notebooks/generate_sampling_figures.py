"""
Given an file of samples write figures visualing samples to an output directory
"""
import click
from pathlib import Path
import matplotlib.pyplot as plt
import pickle as pkl
import _model_variations as mv
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib as mpl

import logging
logger = logging.getLogger(__name__)
# Globals
hist_range = (-1, 1)

@click.command()
@click.option("--o", type=str, help="output directory")
@click.option("--i", type=str, help="input file")
def main(o, i):
    _main(o, i)

_base_style = ""
_corr_matrix_style = ""
_spec_counts_style = ""
_histogram_style = ""
_scatter_plot_style = ""
_caterpillar_style = ""


def rc_context(rc = None, fname = None):
    """
    Decorator maker
    """
    def rc_decorator(func):
        def wrapper(*args, **kwargs):
            with mpl.rc_context(rc = rc, fname = fname):
                return func(*args, **kwargs)
        return wrapper
    return rc_decorator
    
def _main(o, i):

    i = Path(i)
    o = Path(o)

    def save(name):
        plt.savefig(str(o / i.stem) + f"_{name}_300.png", dpi=300)
        plt.savefig(str(o / i.stem) + f"_{name}_1200.png", dpi=1200)
        plt.close()

    def plot_spec_count_table(table, xlabel, ylabel, savename, title):
        fig, ax = plt.subplots()
        plt.matshow(np.array(table))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.colorbar()
        save(savename)
    
    def plot_correlation_matrix(matrix, xlabel, savename, title, ylabel = None):
        if ylabel is None:
            ylabel = xlabel
        fig, ax = plt.subplots()
        plt.matshow(matrix)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    def plot_matrix(matrix, savename):
        fig, ax = plt.subplots()
        plt.matshow(matrix)
        plt.colorbar()
        save(savename)


    def plot_histogram(x, xlabel, ylabel, title, savename, bins = 100, hist_range=(-1, 1),
                       text = None, textx = None, texty = None,): 
        fig, ax = plt.subplots()
        plt.hist(x, bins = bins, range = range)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if text:
            plt.text(textx, texty, text)
        save(savename)

    def plot_caterpillar(y, xlabel, ylabel, title, savename, alpha = 0.2):
        fig, ax = plt.subplots()
        x = np.arange(len(y))
        ax.plot(x, y, alpha = alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    def plot_plot(x, y, fmt, xlabel, ylabel, savename, title = None, alpha = 0.2):
        fig, ax = plt.subplots()
        ax.plot(x, y, fmt, alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)
    
    # Get the model data
    with open(str(i.parent / i.stem) + "_model_data.pkl", "rb") as f:
        model_data = pkl.load(f)
    i_ = i

    # Correlation matrix
    plot_correlation_matrix(
            matrix = model_data['corr'],
            xlabel = 'node',
            savename = 'corr',
            title = "$R$",)

    # Correlation matrix
    plot_correlation_matrix(
            matrix = model_data['shuff_corr'],
            xlabel = "node",
            savename = "shuff_corr",
            title = "$R^{0}$",)

    # Composite dict - skip

    # Matrix plot of shuffled spec table
    plot_spec_count_table(
            table = model_data['selected_shuff_spec_table'],
            xlabel = "condition",
            ylabel = "node",
            savename = "selected_shuff_spec_table",
            title = "Shuffled spectral counts"),

    # Matrix plot of spec table
    plot_spec_count_table(
            table = model_data['selected_spec_table'],
            xlabel = "condition",
            ylabel = "node",
            savename = "selected_spec_table",
            title = "Spectral counts",)

    # lower_edge_prob - skip
    # uppder_edge_prob - skip
    # thresholds - skip

    # Matrix plot of correlation matrix after shuffling 
    plot_correlation_matrix(
            matrix = model_data['shuff_corr_all'],
            xlabel = "node",
            savename = "shuff_corr_all",
            title = "$R^0$",) 
    
    # name2node_idx - skip
    # node_idx2name - skip

    # N skip
    # M skip

    # Histogram of profile correlations 
    plot_histogram(
            x = model_data['apms_corr_flat'],
            title = "AP-MS profile correlations",
            xlabel = "$R_{ij}$",
            ylabel = "Frequency",
            savename = "apms_corr_flat")

    # Histogram of Null all
    n_null_bins = model_data['n_null_bins']
    R0 = model_data['apms_shuff_corr_all_flat']
    plot_histogram(
            x = R0, 
            title = "AP-MS permuted correlations",
            xlabel = "$R_{ij}^{0}$",
            ylabel = "Frequency",
            bins = n_null_bins,
            text = f"N {len(R0)}\nbins : {n_null_bins}",
            textx = 0.1,
            texty = 0.5,
            savename = "apms_shuff_corr_all_flat")

    # Histogram of Null from selected columns 
    R0 = model_data['apms_shuff_corr_flat']
    plot_histogram(
            x = R0, 
            title = "AP-MS permuted correlations",
            xlabel = "$R_{ij}^{0}$",
            ylabel = "Frequency",
            bins = n_null_bins,
            text = f"N {len(R0)}\nbins : {n_null_bins}",
            textx = 0.1,
            texty = 0.5,
            savename = "apms_shuff_corr_flat")

    # Plot the spec_table
    spec_table = pd.read_csv(str(o / "spec_table.tsv"), sep="\t", index_col=0)
    plot_spec_count_table(
            table = spec_table,
            xlabel = "node",
            ylabel = "condition",
            savename = "spec_table",)

    N = model_data['N']
    R = mv.flat2matrix(R, N)
    R = np.array(R)

    # Mean Variance correlation plot
    fig, ax = plt.subplots()
    plt.matshow(R)
    plt.colorbar()
    save("reconstructed_apms_corr")


    with open(i, "rb") as f:
        x = pkl.load(f)
    
    # Unpack
    samples = x['samples']
    ef = x['extra_fields']
    nsamples, M = samples['z'].shape
    N = model_data['N']
    a = jax.nn.sigmoid((samples['z']-0.5)*1_000)
    mean = np.mean(a, axis=0) 

    # Plot a table of the average value of every composite N
    if "new_composite_dict_norm_approx" in model_data:
        save_composite_table(model_data, o, samples)
    # Plot a graph of composite data satisfaction

    # Plot the average adjacency matrix  

    # Plot a histogram of edges
    plot_histogram(
            x = mean,
            xlabel = "Mean edge value",
            ylabel = "Frequency",
            savename = "edge_score_hist",
            hist_range = (0, 1),)

    var = np.var(a, axis=0) 
    A = mv.flat2matrix(mean, N)
    V = mv.flat2matrix(var, N)

    plot_matrix(
            matrix = np.array(A),
            savename = "mean_adjacency",)

    # Energy histogram
    plot_histogram(
            x = np.array(ef["energy"]),
            xlabel = "Energy",
            ylabel = "Frequency",
            title = "E = U + K", 
            savename = "energy",)
    
    x_cater = np.arange(len(ef['energy'])) 
    # Energy Caterpillar
    plot_caterpillar(
            y = np.array(ef['energy']), 
            xlabel = "Post warmup MCMC step",
            ylabel = "Energy",
            title = "E = U + k",
            savename = "energy_caterpill",)

    # Mean acceptance histogram
    plot_histogram(
            x = np.array(ef['mean_accept_prob']),
            xlabel = "Mean acceptance probability",
            ylabel = "Frequency",
            title = None,
            savename = "mean_accept_prob",)

    
    # potential energy 
    pe = np.array(ef['potential_energy'])

    plot_histogram(
            x = pe,
            xlabel = "Potential energy",
            ylabel = "Frequency",
            title = "U",
            savename = "potential_energy",)

    plot_caterpillar(
            y = pe,
            xlabel = "Post warmup MCMC step",
            ylabel = "Potential energy",
            title = "U",
            savename = "potential_energy_caterpill",)
    
    plot_histogram(
            x = np.array(ef['diverging']),
            xlabel = "Diverging",
            ylabel = "Frequency",
            savename = "diverging",)
    
    plot_histogram(
            x = np.array(ef['accept_prob']),
            xlabel = "Acceptance probability",
            ylabel = "Frequency",
            savename = "accept_prob",)

    # Plot the hist
    """
    # Animation 
    fig, ax = plt.subplots()
    mat = ax.matshow(np.zeros((N, N)))
    
    def animate(i):
        a_i = mv.flat2matrix(a[i, :], N)
        mat = ax.matshow(np.array(a_i))
        return mat,

    nsamples, nedges = samples['z'].shape 
    ani = animation.FuncAnimation(fig, animate, repeat=True,
       frames=nsamples, interval=50)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15,
                                     metadata=dict(artist='Me'),
                                     bitrate=1800)

    ani.save('edge_matrix.gif', writer=writer)
    """
    plot_matrix(
            matrix = np.array(V),
            savename = "var_adjacency",)

    # Mean Variance correlation plot
    plot_plot(
            x = mean,
            y = var,
            fmt = 'k.',
            xlabel = "Edge mean",
            ylabel = "Edge variance",
            savename = "mean_var_scatter",)

    # Distribution of pN
    plot_histogram(
            x = pN,
            xlabel = "$u$",
            ylabel = "Frequency",
            title = "Nuisance variable $u$",
            savename = "hist_u",) 

    # Write an edge list table of scores
    w = []
    a = []
    b = []
    node_idx2name = model_data['node_idx2name']

    viral_proteins = ["vifprotein", "tatprotein", "nefprotein", "gagprotein",
                      "polpolyprotein"]

    for i in range(N):
        for j in range(0, i):
            w.append(A[i, j])
            a.append(node_idx2name[i])
            b.append(node_idx2name[j])
    
    df = pd.DataFrame({'a': a, 'b': b, 'w': w})
    base_name = str(o / i_.stem)
    df.to_csv(base_name + "edge_score_table.tsv", sep="\t", index=None)

    # Analysis

    # Count the number of Viral-Viral interactions 

    return samples

def save_composite_table(model_data, o, samples):
    Ns = []
    means = []
    stds = []
    nmax = []
    for i in range(1000):
        key = f"c{i}_N"
        if key in samples:
            Ns.append(key)
            vals = samples[key]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            nmax.append(model_data['new_composite_dict_norm_approx'][i]['N'])
    df = pd.DataFrame({"Nmax" : nmax, "av" : means, "std" : stds}, index=Ns) 
    outpath = str(o / "composite_N.tsv")
    df.to_csv(outpath, sep="\t")

_example = {"i" : "../cullin_run0/0_model23_ll_lp_13.pkl", "o": "../cullin_run0/"}

if __name__ == "__main__":
    main()
