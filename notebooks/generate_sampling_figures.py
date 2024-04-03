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

# Globals
hist_range = (-1, 1)

@click.command()
@click.option("--o", type=str, help="output directory")
@click.option("--i", type=str, help="input file")
def main(o, i):
    _main(o, i)

def _main(o, i):
    with open(i, "rb") as f:
        x = pkl.load(f)

    samples = x['samples']
    ef = x['extra_fields']

    i = Path(i)
    o = Path(o)

    def save(name):
        plt.savefig(str(o / i.stem) + f"_{name}_300.png", dpi=300)
        plt.savefig(str(o / i.stem) + f"_{name}_1200.png", dpi=1200)
        plt.close()

    # Energy histogram
    fig, ax = plt.subplots()
    ax.hist(np.array(ef['energy']))
    ax.set_xlabel("Energy")
    ax.set_ylabel("Frequency")
    save("energy")

    x_cater = np.arange(len(ef['energy'])) 

    # Energy Caterpillar
    fig, ax = plt.subplots()
    ax.plot(x_cater, np.array(ef['energy']), alpha=0.2)
    ax.set_xlabel("Post warmup step")
    ax.set_ylabel("Energy")
    save("energy_caterpill")

    # Mean acceptance histogram
    fig, ax = plt.subplots()
    ax.hist(np.array(ef['mean_accept_prob']))
    ax.set_xlabel("mean_accept_prob")
    ax.set_ylabel("Frequency")
    save("mean_accept_prob")

    # potential energy 
    fig, ax = plt.subplots()
    ax.hist(np.array(ef['potential_energy']))
    ax.set_xlabel("potential_energy")
    ax.set_ylabel("Frequency")
    save("potential_energy")

    # potential 
    fig, ax = plt.subplots()
    ax.plot(x_cater, np.array(ef['potential_energy']), alpha=0.2)
    ax.set_xlabel("Post warmup step")
    ax.set_ylabel("potential_energy")
    save("potential_energy_caterpill")

    # diverging 
    fig, ax = plt.subplots()
    ax.hist(np.array(ef['diverging']))
    ax.set_xlabel("diverging")
    ax.set_ylabel("Frequency")
    save("diverging")

    # accept_prob 
    fig, ax = plt.subplots()
    ax.hist(np.array(ef['accept_prob']))
    ax.set_xlabel("accept_prob")
    ax.set_ylabel("Frequency")
    save("accept_prob")

    # Get the model data
    with open(str(i.parent / i.stem) + "_model_data.pkl", "rb") as f:
        model_data = pkl.load(f)
    i_ = i

    # Plot the average adjacency matrix  
    nsamples, M = samples['z'].shape
    N = model_data['N']
    a = jax.nn.sigmoid((samples['z']-0.5)*1_000)
    mean = np.mean(a, axis=0) 
    # Plot a histogram of edges
    fig, ax = plt.subplots()
    plt.hist(mean, bins=100, range=(0, 1))
    plt.xlabel("Mean Edge Value")
    plt.ylabel("Frequency") 
    save("edge_score_hist")
    var = np.var(a, axis=0) 
    A = mv.flat2matrix(mean, N)
    V = mv.flat2matrix(var, N)
    fig, ax = plt.subplots()
    plt.matshow(np.array(A))
    plt.colorbar()
    save("mean_adjacency")

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

    # plot variance
    fig, ax = plt.subplots()
    plt.matshow(np.array(V))
    plt.colorbar()
    save("var_adjacency")

    # Mean Variance correlation plot
    fig, ax = plt.subplots()
    ax.plot(mean, var, 'k.', alpha=0.2)
    ax.set_xlabel("Edge mean")
    ax.set_ylabel("Edge var")
    save("mean_var_scatter")

    # Plot the Profile similarity matrix of input information 
    data_path = str(i.parent)
    model_data = mv.model23_ll_lp_data_getter(data_path) 
    model_data = mv.model23_data_transformer(model_data)

    # Histogram of Model Inputs 
    R = model_data['apms_corr_flat']
    fig, ax = plt.subplots()
    plt.hist(R, bins=100, range=hist_range)
    ax.set_xlabel("$R_{ij}$")
    ax.set_ylabel("Frequency")
    save("Rhist")
    
    # Histogram of Null all
    R0 = model_data['apms_shuff_corr_all_flat']
    fig, ax = plt.subplots()
    plt.hist(R0, bins=100, range=hist_range)
    plt.title("All Null Correlations")
    ax.set_xlabel("$R_{ij}^{0}$")
    ax.set_ylabel("Frequency")
    save("Rhist_null_all")

    # Histogram of Null from selected columns 
    R0 = model_data['apms_shuff_corr_flat']
    fig, ax = plt.subplots()
    plt.hist(R0, bins=100, range=hist_range)
    plt.title("Masked null correlations")
    ax.set_xlabel("$R_{ij}^{0}$")
    ax.set_ylabel("Frequency")
    save("Rhist_null_mask")

    # Matrix plot of shuffled spec table
    shuff_spec_table = model_data['shuff_spec_table']
    fig, ax = plt.subplots()
    plt.matshow(np.array(shuff_spec_table))
    ax.set_xlabel("$Conditions$")
    ax.set_ylabel("Node")
    save("ShuffledSpecTable")

    N = model_data['N']
    R = mv.flat2matrix(R, N)
    R = np.array(R)

    # Mean Variance correlation plot
    fig, ax = plt.subplots()
    plt.matshow(R)
    plt.colorbar()
    save("PearsonR")


    # Plot a table of the average value of every composite N
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
    # Plot a graph of composite data satisfaction

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

_example = {"i" : "../cullin_run0/0_model23_ll_lp_13.pkl", "o": "../cullin_run0/"}

if __name__ == "__main__":
    main()
