"""
Given an file of samples write figures visualing samples to an output directory
"""
import click
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import pickle as pkl
import _model_variations as mv
import data_io
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
@click.option("--mode", type=str, default="cullin")
def main(o, i, mode):
    _main(o, i, mode)

_base_style = ""
_corr_rc =   {"image.cmap" : "coolwarm"} 
_matrix_rc = {"image.cmap" : "hot"}
_spec_rc =   {"image.cmap" : "gist_gray"} 
_histogram_rc = ""
_scatter_plot_rc = ""
_caterpillar_rc = ""

_dev = True

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

def input_load(i, fbasename, suffix):
    with open(i / f"{fbasename}{suffix}.pkl", "rb") as f:
        return pkl.load(f)


def n_tp(aij, refij):
    """
    Return the values of the array where both are 1
    Boolean arrays
    """
    return np.sum((aij == 1) &  (refij == 1))

def n_fp(aij, refij):
    return np.sum((aij == 1) & (refij == 0))

def n_edge(aij):
    return np.sum(aij > 0.5)

def _loop(aij_mat, refij, f):
    N, _ = aij_mat.shape
    output = np.zeros(N)
    for i in range(N):
        output[i] = f(aij_mat[i, :], refij)
    return output

n_tps = partial(_loop, f=n_tp)
n_fps = partial(_loop, f=n_fp)
n_edges = partial(_loop, f=lambda x, _: n_edge(x), refij=None)

def n_tps_from_samples(samples, refij):
    aij_mat = mv.Z2A(samples['z'])
    aij_mat = aij_mat > 0.5
    return n_tps(aij_mat, refij)

def n_fps_from_samples(samples, refij):
    aij_mat = mv.Z2A(samples['z'])
    aij_mat = aij_mat > 0.5
    return n_fps(aij_mat, refij)

def postprocess_samples(i, fbasename):
    # Get the model data
    warmup_out_dict = input_load(i, fbasename, "_warmup_samples") 
    sample_out_dict = input_load(i, fbasename, "")

    def concat_warmup_samples(w, s):
        ws = w['samples']
        ss = s['samples']

        we = w['extra_fields']
        se = s['extra_fields']

        # concatenate along the first axis with warmup first 
        os = {}
        oe = {}
        found_keys = []
        for key in ws:
            os[key] = np.concatenate([ws[key], ss[key]])
            found_keys.append(key)
        for key in we:
            oe[key] = np.concatenate([we[key], se[key]])
            found_keys.append(key)
        found_keys = list(set(found_keys))

        out = {"samples" : os, "extra_fields" : oe}

        # add keys not in warmup
        for key in se:
            if key not in found_keys:
                out['extra_fields'][key] = se[key]
        for key in ss:
            if key not in found_keys:
                out['samples'][key] = ss[key]
        for key in s:
            if key not in found_keys:
                if key not in ("samples", "extra_fields"):
                    out[key] = s[key]
        return out 

    results = concat_warmup_samples(warmup_out_dict, sample_out_dict) 
    return results


def align_reference_to_model(model_data, uref, mode="cullin",):
    """
    Order edges in the reference according to the model.
    """
    n_possible_edges = model_data['M']
    node_idx2name = model_data["node_idx2name"]
    n_nodes = model_data['N']
    # uref is in UID but model is in names. Reindex
    if mode == "cullin":
        reindexer = {val : key for key, val in data_io.get_cullin_reindexer().items()}
        uref.reindex(reindexer, enforce_coverage = False) 
    else:
        raise NotImplementedError("Only cullin has been implemented")

    uref._build_edge_dict()
    reflist_out = np.zeros(n_possible_edges, dtype=np.int64)
    for k in range(n_possible_edges):
        i, j = mv.ij_from(k, n_nodes)
        i_node_name = node_idx2name[i]
        j_node_name = node_idx2name[j]
        edge = frozenset([i_node_name, j_node_name])
        if edge in uref._edge_dict:
            reflist_out[k] = 1
        k += 1
    assert j == n_nodes -1
    return reflist_out

    
def _main(o, i, mode):
    logger = logging.getLogger(__name__)
    logger.info("Enter generate_sampling_figures")
    logging.info(f"Params")
    logging.info(f"    i:{i}")
    logging.info(f"    o:{o}")
    logging.info(f" mode:{mode}")

    i = Path(i)
    o = Path(o)

    def save(name):
        plt.savefig(str(o / i.stem) + f"_{name}_300.png", dpi=300)
        plt.savefig(str(o / i.stem) + f"_{name}_1200.png", dpi=1200)
        plt.close()
    
    @rc_context(rc = _spec_rc)
    def plot_spec_count_table(table, xlabel, ylabel, savename, title):
        fig, ax = plt.subplots()
        plt.matshow(np.array(table))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.colorbar()
        save(savename)

    @rc_context(rc = _corr_rc) 
    def plot_correlation_matrix(matrix, xlabel, savename, title, ylabel = None):
        if ylabel is None:
            ylabel = xlabel
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, vmin=-1, vmax=1)
        plt.colorbar(shrink = 0.95, mappable = cax, ax = ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    @rc_context(rc = _matrix_rc) 
    def plot_matrix(matrix, xlabel, savename, title, ylabel = None):
        if ylabel is None:
            ylabel = xlabel
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, vmin=0, vmax=1)
        plt.colorbar(shrink = 0.95, mappable = cax, ax = ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    @rc_context(rc = _matrix_rc)
    def plot_binary_matrix(matrix):
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, vmin=0, vmax=1)
        plt.colorbar(shrink = 0.95, mappable = cax, ax = ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)


    def plot_histogram(x, xlabel, ylabel, savename, title = None, bins = 100, hist_range=(-1, 1),
                       text = None, textx = None, texty = None,): 
        if title is None:
            N = len(x)
            title = f"N={N}    {bins} bins"
        fig, ax = plt.subplots()
        plt.hist(x, bins = bins, range = hist_range)
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

    def model_data_plots():
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
                ylabel = "count",
                savename = "apms_corr_flat")
    
        # Histogram of Null all
        n_null_bins = model_data['n_null_bins']
        R0 = model_data['apms_shuff_corr_all_flat']
        plot_histogram(
                x = R0, 
                title = "AP-MS permuted correlations",
                xlabel = "$R_{ij}^{0}$",
                ylabel = "count",
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
                ylabel = "count",
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
                title = "Spectral counts",
                savename = "spec_table",)

        # Conditionally plot the saint pair score edgelist table
        conditional_keys = ("saint_max_pair_score_edgelist",)
        for cond_key in conditional_keys:
            if cond_key in model_data:
                cond_val = model_data[cond_key]
                if cond_val == "saint_max_pair_score_edgelist":
                    cond_mat = mv.flat2matrix(cond_val, n=N)
                    plot_matrix(cond_mat,
                                xlabel = "node",
                                title = "Saint max pair score",
                                savename = "saint_max_pair_score")


    # Get the model data
    with open(str(i.parent / i.stem) + "_model_data.pkl", "rb") as f:
        model_data = pkl.load(f)
    i_ = i
    
    if not _dev:
        model_data_plots()
    # Get concatenated samples

    direct_ref = data_io.get_pdb_ppi_predict_direct_reference()
    costructure_ref = data_io.get_pdb_ppi_predict_cocomplex_reference()
    

    direct_ij = align_reference_to_model(model_data, direct_ref, mode=mode) 
    costructure_ij = align_reference_to_model(model_data, costructure_ref, mode=mode)

    rng_key = jax.random.PRNGKey(122387)
    shuff_direct_ij = np.array(jax.random.permutation(rng_key, direct_ij))


    x : dict = postprocess_samples(i.parent, fbasename=i.stem) 
    with open(i, "rb") as f:
        x2 = pkl.load(f)

    N = model_data['N']
    def run_plots(x, suffix=""):
        # Unpack
        samples = x['samples']
        ef = x['extra_fields']
        nsamples, M = samples['z'].shape
        a = jax.nn.sigmoid((samples['z']-0.5)*1_000)
        mean = np.mean(a, axis=0) 
        a_typed = np.array(a > 0.5)

        control_ij = a_typed[0, :]
        control_last_ij = a_typed[-1, :]

        direct_tps = np.array(n_tps(a_typed, direct_ij))
        direct_fps = np.array(n_fps(a_typed, direct_ij))

        costructure_tps = np.array(n_tps(a_typed, costructure_ij))
        costructure_fps = np.array(n_fps(a_typed, costructure_ij))

        control_first_tps = np.array(n_tps(a_typed, control_ij))
        control_first_fps = np.array(n_fps(a_typed, control_ij))

        control_last_tps = np.array(n_tps(a_typed, control_last_ij))
        control_last_fps = np.array(n_fps(a_typed, control_last_ij))

        shuff_direct_tps = np.array(n_tps(a_typed, shuff_direct_ij))
        shuff_direct_fps = np.array(n_fps(a_typed, shuff_direct_ij))
        
        n_total_edges = np.array(n_edges(a_typed))
        score = np.array(ef['potential_energy'])
        
        plot_plot(direct_tps,       score, "k.", "direct tp edges",       "score", "direct_tp_vs_score" + suffix) 
        plot_plot(direct_fps,       score, "k.", "direct fp edges",       "score", "direct_fp_vs_score" + suffix) 
        
        plot_plot(costructure_tps,  score, "k.", "costructure tp edges",  "score", "costructure_tp_vs_score" + suffix) 
        plot_plot(costructure_fps,  score, "k.", "costructure fp edges",  "score", "costructure_fp_vs_score" + suffix) 

        plot_plot(control_first_tps,      score, "k.", "CONTROL FIRST position tp edges", "score", "control_tp_vs_score" + suffix) 
        plot_plot(control_first_fps,      score, "k.", "CONTROL FIRST position fp edges", "score", "control_fp_vs_score" + suffix) 

        plot_plot(control_last_tps,      score, "k.", "CONTROL LAST position tp edges", "score", "control_last_tp_vs_score" + suffix) 
        plot_plot(control_last_fps,      score, "k.", "CONTROL LAST position fp edges", "score", "control_last_fp_vs_score" + suffix) 

        plot_plot(shuff_direct_tps, score, "k.", "shuff direct tp edges", "score", "shuff_direct_tp_vs_score" + suffix) 
        plot_plot(shuff_direct_fps, score, "k.", "shuff direct fp edges", "score", "shuff_direct_fp_vs_score" + suffix) 

        plot_plot(n_total_edges,    score, "k.", "N edges", "score", "n_edges_vs_score" + suffix)

        for key, val in dict(
                direct_tps = direct_tps,
                direct_fps = direct_fps,
                costructure_tps = costructure_tps,
                costructure_fps = costructure_fps,
                control_first_tps = control_first_tps,
                control_first_fps = control_first_fps,
                control_last_tps = control_last_tps,
                control_last_fps = control_last_fps,
                shuff_direct_tps = shuff_direct_tps,
                shuff_direct_fps = shuff_direct_fps,
                n_total_edges = n_total_edges,
                ).items():
            plot_caterpillar(
                y = val,
                xlabel = "iteration",
                ylabel = key,
                title = key,
                savename = f"{key}_caterpill" + suffix,)


        # Plot a table of the average value of every composite N
        if "new_composite_dict_norm_approx" in model_data:
            save_composite_table(model_data, o, samples)
        # Plot a graph of composite data satisfaction

        # Plot the average adjacency matrix  

        # Plot a histogram of edges

        plot_histogram(
                x = mean,
                xlabel = "mean edge value",
                ylabel = "count",
                savename = "edge_score_hist" + suffix,
                hist_range = (0, 1),)

        var = np.var(a, axis=0) 
        A = mv.flat2matrix(mean, N)
        V = mv.flat2matrix(var, N)
        plot_matrix(
                matrix = mv.flat2matrix(a[0, :] , N), 
                xlabel = "node",
                title = "Initial value",
                savename = "initial_value" + suffix,
                )
        plot_matrix(
                matrix = np.array(A),
                xlabel = "node",
                title = "Average edge score",
                savename = "mean_adjacency" + suffix,)

        # Energy histogram
        energy = np.array(ef["energy"])
        plot_histogram(
                x = energy,
                xlabel = "energy",
                ylabel = "count",
                title = "H(q,p) = U(q) + K(p)", 
                hist_range = None,
                savename = "energy" + suffix,)
        
        x_cater = np.arange(len(ef['energy'])) 
        # Energy Caterpillar
        plot_caterpillar(
                y = energy, 
                xlabel = "iteration",
                ylabel = "energy",
                title = "H(q,p) = U(q) + K(p)",
                savename = "energy_caterpill" + suffix,)

        # Mean acceptance histogram
        plot_histogram(
                x = np.array(ef['mean_accept_prob']),
                xlabel = "mean acceptance probability",
                ylabel = "count",
                title = None,
                hist_range = (0.5, 1.),
                savename = "mean_accept_prob" + suffix,)

        # potential energy 
        pe = np.array(ef['potential_energy'])

        plot_histogram(
                x = pe,
                xlabel = "potential energy",
                ylabel = "count",
                title = "U",
                hist_range = None,
                savename = "potential_energy" + suffix,)

        plot_caterpillar(
                y = pe,
                xlabel = "iteration",
                ylabel = "potential energy",
                title = "Score",
                savename = "potential_energy_caterpill" + suffix,)

        ke = energy - pe
        plot_histogram(
                x = ke,
                xlabel = "kinetic energy",
                ylabel = "count",
                title = "k",
                hist_range = None,
                savename = "kinetic_energy" + suffix,)
        
        plot_caterpillar(
                y = ke,
                xlabel = "iteration",
                ylabel = "kinetic energy",
                title = "k",
                savename = "kinetic_energy_caterpill" + suffix,)
        
        plot_histogram(
                x = np.array(ef['diverging']),
                xlabel = "diverging",
                ylabel = "count",
                savename = "diverging" + suffix,)
        
        plot_histogram(
                x = np.array(ef['accept_prob']),
                xlabel = "acceptance probability",
                ylabel = "count",
                savename = "accept_prob" + suffix,)

        for key in ("accept_prob", "mean_accept_prob", "diverging", "num_steps",
                    "r_score", "sij_score", "z_score"):
            plot_stuff = False
            if (key in ef):  
                d = ef
                plot_stuff = True
            elif (key in samples):
                d = samples
                plot_stuff = True
            
            if plot_stuff:
                val = d[key]
                plot_caterpillar(
                        y = val,
                        xlabel = "iteration",
                        ylabel = key,
                        title = key,
                        savename = f"{key}_caterpill" + suffix,)

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
                title = "Edge variance",
                xlabel = "node",
                savename = "var_adjacency" + suffix,)

        # Mean Variance correlation plot
        plot_plot(
                x = mean,
                y = var,
                fmt = 'k.',
                xlabel = "Edge mean",
                ylabel = "Edge variance",
                savename = "mean_var_scatter" + suffix,)

        # Distribution of u 
        if 'u' in samples:
            u = samples['u']
            plot_histogram(
                    x = u,
                    xlabel = "$u$",
                    ylabel = "count",
                    title = "Nuisance variable $u$",
                    hist_range = None, #(0, 0.005),
                    savename = "hist_u" + suffix,) 
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
        df.to_csv(base_name + f"edge_score_table{suffix}.tsv", sep="\t", index=None)

    run_plots(x, suffix="_w_warmup")
    run_plots(x2)
    model_data_plots()
    # Write an edge list table of scores

    # Analysis

    # Count the number of Viral-Viral interactions 

    #return samples


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
