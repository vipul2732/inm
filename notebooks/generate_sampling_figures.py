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

    # Plot the average adjacency matrix  
    nsamples, M = samples['z'].shape
    N = model_data['N']
    a = jax.nn.sigmoid((samples['z']-0.5)*1_000)
    mean = np.mean(a, axis=0) 
    var = np.var(a, axis=0) 
    A = mv.flat2matrix(mean, N)
    V = mv.flat2matrix(var, N)
    fig, ax = plt.subplots()
    plt.matshow(np.array(A))
    plt.colorbar()
    save("mean_adjacency")
    
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

    return samples

_example = {"i" : "../cullin_run0/0_model23_ll_lp_13.pkl", "o": "../cullin_run0/"}

if __name__ == "__main__":
    main()
