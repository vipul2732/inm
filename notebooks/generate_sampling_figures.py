"""
Given an file of samples write figures visualing samples to an output directory
"""
import click
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib as mpl

import logging

import data_io
import _model_variations as mv
import merge_analysis as merge 

import scipy as sp

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

def run_multichain_specific_plots(x, model_data, suffix="", save=None, o = None):
    """
    Params:
      x : { 
      CHAIN       
      }
    Types of plots to run

    Iteration scatter w/ chain std.
    """
    def caterrplot(y, xlabel, ylabel, title, savename, alpha = 0.2, save=save):
        fig, ax = plt.subplots()
        nchains, niter = y.shape
        x = np.arange(niter)
        ax.errorbar(x, np.mean(y, axis=0), yerr=np.std(y, axis=0), fmt='o', alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)
    merged_results_two_subsets_scores_hist_and_test(x, save=save, o = o)   

    # Plot of improving scores

    ef = x["extra_fields"]
    logging.info("calculating best score per chain based on amount of sampling")
    results_dict = best_score_per_chain_based_on_amount_of_sampling(ef["potential_energy"])
    errplot_from_av_std_dict(results_dict, "amount of sampling", "best score", "Best score per chain", "best_score_per_chain" + suffix, save=save)

    pdb_ppi_direct = data_io.get_pdb_ppi_predict_direct_reference()
    direct_ij = align_reference_to_model(model_data, pdb_ppi_direct, mode="cullin")

    score_vs_ppv_plot(jax.random.PRNGKey(0), x["samples"], ef, direct_ij, save=save, o=o)

    logging.info("calculating ppv per chain based on amount of sampling")
    ppv_results_dict = ppv_per_chain_based_on_amount_of_sampling(x['samples']['z'] > 0.5, direct_ij, amount_of_sampling_list = None) 
    errplot_from_av_std_dict(ppv_results_dict, "amount of sampling", "ppv", "PPV per chain", "ppv_per_chain" + suffix, save=save)
    top_ppv_dict = top_ppv_per_chain_based_on_amount_of_sampling(x['samples']['z'] > 0.5, direct_ij, amount_of_sampling_list = None)
    errplot_from_av_std_dict(top_ppv_dict, "amount of sampling", "top ppv", "Top PPV per chain", "top_ppv_per_chain" + suffix, save=save)


    for key in ef:
        val = ef[key]
        caterrplot(val, "iteration", key, key, f"{key}_caterpill" + suffix)    
    
def ppv(aij, refij):
    """
    Positive predictive value
    """
    n_tp = jnp.sum((aij == 1) &  (refij == 1))
    n_fp = jnp.sum((aij == 1) & (refij == 0))
    return n_tp / (n_tp + n_fp)

def ppv_per_iteration(aij_mat, refij):
    """
    Return the positive predictive value for each iteration
    """
    N, _ = aij_mat.shape
    output = jnp.zeros(N)
    for i in range(N):
        output = output.at[i].set(ppv(aij_mat[i, :], refij))
    return output

def ppv_per_iteration_vectorized(aij_mat, refij):
    ppv_vectorized = jax.vmap(ppv, in_axes=(0, None))
    output = ppv_vectorized(aij_mat, refij)
    #breakpoint()
    return output


def min_score_vectorized(score_mat):
    """
    score_mat : (chain, iteration) 
    """
    return jnp.min(score_mat, axis=1)

def metric_as_a_function_of_amount_of_sampling_per_chain(x, metricf, amount_of_sampling_list):
    """
    x : (CHAIN, ITERATION, ...) 
    metricf : 
    """
    M = len(amount_of_sampling_list)
    out = jnp.zeros((M, 2))
    for i, N in enumerate(amount_of_sampling_list):
        temp = x[:, 0:N, ...] # select up to N iterations per chain
        metric = metricf(temp)
        if metric.ndim > 0:
            av = jnp.mean(metric, axis=0) # average over chains
            std = jnp.std(metric, axis=0) # std over chains
        else:
            av = metric
            std = 0
        out = out.at[i, :].set([av, std])
    return dict(out = out,
                amount_of_sampling_list = amount_of_sampling_list)


_STD_AMOUNT_OF_SAMPING = [1] + list(np.arange(10, 2_000, 10)) 
_STD_AMOUNT_OF_SAMPLING2 = [1,10, 100, 500,] + list(np.arange(1_000, 20_000, 1_000))
_CHAIN_DIM = 0
_ITER_DIM = 1
_AV_DIM = 0
_STD_DIM = 1

def best_score_per_chain_based_on_amount_of_sampling(
        x, amount_of_sampling_list = None):
    if x.ndim == 2:
        n_chains, n_iter = x.shape
    elif x.ndim == 3:
        n_chains, n_iter, _ = x.shape
    else:
        raise ValueError("Only 2 or 3 dimensions are supported")
    if amount_of_sampling_list is None:
        if n_iter > 2_000:
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPLING2
        else: 
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPING 
    results = metric_as_a_function_of_amount_of_sampling_per_chain(x, lambda x: jnp.min(x, axis=_ITER_DIM), amount_of_sampling_list)
    return results

def top_ppv_per_chain_based_on_amount_of_sampling(
        x, refij, amount_of_sampling_list = None):
    nchains, niter, vdim = x.shape
    if amount_of_sampling_list is None:
        if niter > 2_000:
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPLING2
        else:
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPING
    results = metric_as_a_function_of_amount_of_sampling_per_chain(x, lambda x: dim_aware_max(ppv_per_iteration_vectorized(x, refij)), amount_of_sampling_list)
    return results

def dim_aware_max(x):
    if x.ndim == 1:
        return jnp.max(x)
    elif x.ndim == 2:
        return jnp.max(x, axis=1)
    else:
        raise ValueError("Only 1 or 2 dimensions are supported")

def ppv_per_chain_based_on_amount_of_sampling(
        x, refij, amount_of_sampling_list = None):
    if x.ndim == 2:
        n_chains, n_iter = x.shape
    elif x.ndim == 3:
        n_chains, n_iter, _ = x.shape
    if amount_of_sampling_list is None:
        if n_iter > 2_000:
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPLING2
        else:
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPING
    results = metric_as_a_function_of_amount_of_sampling_per_chain(x, lambda x: ppv_per_iteration_vectorized(x, refij), amount_of_sampling_list)
    return results

def score_vs_ppv_plot(rng_key, samples, ef, refij, save=None, o=None, N=10_000):
    """
    Plot the score vs ppv for each chain
    """

    scores = ef["potential_energy"]
    assert scores.ndim == 2
    n_chains, n_iter = scores.shape
    
    Nsamples = n_chains * n_iter

    indices = jax.random.permutation(rng_key, jnp.arange(Nsamples))[:N]

    scores = jnp.ravel(scores)[indices]
    aij_mat = mv.Z2A(samples['z']) > 0.5
    ppv = ppv_per_iteration_vectorized(aij_mat, refij)
    breakpoint()



def errplot_from_av_std_dict(av_std_dict, xlabel, ylabel, title, savename, save=None, alpha=0.2):
    fig, ax = plt.subplots()
    x = av_std_dict['amount_of_sampling_list']
    y = av_std_dict['out']
    ax.errorbar(x, y[:, _AV_DIM], yerr=y[:, _STD_DIM], fmt='.', alpha=alpha, capsize=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save(savename)
    


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

def merged_results_two_subsets_scores_hist_and_test(merged_results, save=None, o = None):
    scores = merged_results["extra_fields"]["potential_energy"]
    
    nchains, niter = scores.shape
    P = nchains // 2
    scores1 = np.ravel(scores[:P, :])
    scores2 = np.ravel(scores[P:, :])
    assert not np.all(scores1 == scores2)

    test_results = sp.stats.ks_2samp(scores1, scores2)

    # write a csv file for the two sample test results
    pd.DataFrame(test_results._asdict(), index = [0],).to_csv(o / "ks_2samp_scores_results.tsv", sep="\t")

    fig, ax = plt.subplots()
    ax.hist(scores1, bins=100, alpha=0.5, label="subset 1")
    ax.hist(scores2, bins=100, alpha=0.5, label="subset 2")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    plt.legend()
    D = "{:.3f}".format(test_results.statistic)
    pval = "{:.3f}".format(test_results.pvalue)

    s = f"Two-sample KS test\nD={D}\np-value={pval}"
    plt.text(0.6, 0.6, s, transform=plt.gca().transAxes)
    save("scores_2_hist")



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

def postprocess_samples(i, fbasename, merge=False):
    """
    Merge argument
    - For a single chain - the concatentation runs along the 0th axis
      concatentating the warmup and non warmup samples.
      (SAMPLES, VARIABLES ...)
    - For multiple chains - the concatenation runs along the 1st axis
      because the 0th axis is the chain axis. 
    """
    if merge:
        def concatf(a, b):
            return np.concatenate([a, b], axis=1)
    else:
        def concatf(a, b):
            return np.concatenate([a, b], axis=0)

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
            os[key] = concatf(ws[key], ss[key])
            found_keys.append(key)
        for key in we:
            oe[key] = concatf(we[key], se[key])
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

def run_multi_chain_version_of_single_chain_plots(x, model_data, save, o, suffix=""):
    # Unpack
    samples = x['samples']
    ef = x['extra_fields']

    nchains, niter, M = samples['z'].shape
    a = mv.Z2A(samples['z']) 
    mean = np.mean(a, axis=0) 

    a_typed = np.array(a > 0.5)

    # generate controls for a single chain
    control_ij = a_typed[0, 0, :] 
    control_last_ij = a_typed[0, -1, :]

    #direct_tps = np.array(n_tps(a_typed, direct_ij))
    #direct_fps = np.array(n_fps(a_typed, direct_ij))

    rng_key = jax.random.PRNGKey(122387)
    direct_bootstrap_reference = bootstrap_samples(rng_key, direct_ij)
    n_boostraps, n_true = direct_bootstrap_reference.shape

    prev_bootstrap_direct_tps = None 
    for i in range(n_boostraps):
        bootstrap_direct_tps = np.array(n_tps(a_typed, direct_bootstrap_reference[i, :]))
        if prev_bootstrap_direct_tps is None:
            prev_bootstrap_direct_tps = bootstrap_direct_tps
        else:
            prev_bootstrap_direct_tps = np.concatenate([prev_bootstrap_direct_tps, bootstrap_direct_tps])

    prev_bootstrap_direct_tps = np.ravel(prev_bootstrap_direct_tps) 
    direct_mu_tps = np.mean(direct_tps)
    direct_sigma_tps = np.mean(direct_tps)

    boot_mu_tps = np.mean(prev_bootstrap_direct_tps)
    boot_mu_sigma_tps = np.std(prev_bootstrap_direct_tps)

    # Make an error bar plot
    fig, ax = plt.subplots()
    x = [0, 1]
    y = [direct_mu_tps, boot_mu_tps]
    yerr = [direct_sigma_tps, boot_mu_sigma_tps]
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    ax.set_title("Quantification of direct TPs vs bootstrapped reference")
    ax.set_ylabel("TPs")
    labels = ["direct", "bootstrap"]
    plt.xticks(x, labels)
    save("direct_vs_bootstrap_tps")

    plt.hist(prev_bootstrap_direct_tps, bins = 100, alpha=0.5, density=True, label="bootstrap direct TPs")
    plt.hist(direct_tps, alpha=0.5, bins = 100, density=True, label="direct TPs")
    plt.xlabel("direct TPs")
    plt.ylabel("density")
    plt.legend()
    save("direct_vs_bootstrap_tps_hist")

    costructure_tps = np.array(n_tps(a_typed, costructure_ij))
    costructure_fps = np.array(n_fps(a_typed, costructure_ij))

    control_first_tps = np.array(n_tps(a_typed, control_ij))
    control_first_fps = np.array(n_fps(a_typed, control_ij))

    control_last_tps = np.array(n_tps(a_typed, control_last_ij))
    control_last_fps = np.array(n_fps(a_typed, control_last_ij))

    shuff_direct_tps = np.array(n_tps(a_typed, shuff_direct_ij))
    shuff_direct_fps = np.array(n_fps(a_typed, shuff_direct_ij))

    shuff_costructure_tps = np.array(n_tps(a_typed, shuff_costructure_ij))
    shuff_costructure_fps = np.array(n_fps(a_typed, shuff_costructure_ij))
    
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

    plot_plot(shuff_costructure_tps, score, "k.", "shuff costructure tp edges", "score", "shuff_costructure_tp_vs_score" + suffix) 
    plot_plot(shuff_costructure_fps, score, "k.", "shuff costructure fp edges", "score", "shuff_costructure_fp_vs_score" + suffix) 

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
            shuff_costructure_tps = shuff_costructure_tps,
            shuff_costructure_fps = shuff_costructure_fps,
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
                "r_score", "sij_score", "z_score", "n_edges_score", "grad_r_score", "grad_z_score"):
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


    
def _main(o, i, mode, merge = False):
    if merge:
        i = Path(str(i.parent) + "_merged") / "merged_results" 
        o = i.parent
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

    def model_data_plots(model_data):
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

    def bootstrap_samples(rng_key, refij, n_boostraps = 100):
        n = refij.shape[0]
        output = jnp.zeros((n_boostraps, n))
        for i in range(n_boostraps):
            rng_key, key = jax.random.split(rng_key)
            shuff_ij = np.array(jax.random.permutation(key, refij))
            output = output.at[i, :].set(shuff_ij)
        return output

    def single_chain_sampling_analysis():

        # Get the model data
        with open(str(i.parent / i.stem) + "_model_data.pkl", "rb") as f:
            model_data = pkl.load(f)
        i_ = i
    
        if not _dev:
            model_data_plots()
        # Get concatenated samples

        direct_ref = data_io.get_pdb_ppi_predict_direct_reference()
        costructure_ref = data_io.get_pdb_ppi_predict_cocomplex_reference()

        rng_key = jax.random.PRNGKey(122387)
        direct_ij = align_reference_to_model(model_data, direct_ref, mode=mode) 
        costructure_ij = align_reference_to_model(model_data, costructure_ref, mode=mode)

        rng_key, key  = jax.random.split(rng_key)
        shuff_costructure_ij = np.array(jax.random.permutation(key, costructure_ij))

        rng_key, key  = jax.random.split(rng_key)
        shuff_direct_ij = np.array(jax.random.permutation(key, direct_ij))

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

            rng_key = jax.random.PRNGKey(122387)
            direct_bootstrap_reference = bootstrap_samples(rng_key, direct_ij)
            n_boostraps, n_true = direct_bootstrap_reference.shape

            prev_bootstrap_direct_tps = None 
            for i in range(n_boostraps):
                bootstrap_direct_tps = np.array(n_tps(a_typed, direct_bootstrap_reference[i, :]))
                if prev_bootstrap_direct_tps is None:
                    prev_bootstrap_direct_tps = bootstrap_direct_tps
                else:
                    prev_bootstrap_direct_tps = np.concatenate([prev_bootstrap_direct_tps, bootstrap_direct_tps])

            prev_bootstrap_direct_tps = np.ravel(prev_bootstrap_direct_tps) 
            direct_mu_tps = np.mean(direct_tps)
            direct_sigma_tps = np.mean(direct_tps)

            boot_mu_tps = np.mean(prev_bootstrap_direct_tps)
            boot_mu_sigma_tps = np.std(prev_bootstrap_direct_tps)

            # Make an error bar plot
            fig, ax = plt.subplots()
            x = [0, 1]
            y = [direct_mu_tps, boot_mu_tps]
            yerr = [direct_sigma_tps, boot_mu_sigma_tps]
            ax.errorbar(x, y, yerr=yerr, fmt='o')
            ax.set_title("Quantification of direct TPs vs bootstrapped reference")
            ax.set_ylabel("TPs")
            labels = ["direct", "bootstrap"]
            plt.xticks(x, labels)
            save("direct_vs_bootstrap_tps")

            plt.hist(prev_bootstrap_direct_tps, bins = 100, alpha=0.5, density=True, label="bootstrap direct TPs")
            plt.hist(direct_tps, alpha=0.5, bins = 100, density=True, label="direct TPs")
            plt.xlabel("direct TPs")
            plt.ylabel("density")
            plt.legend()
            save("direct_vs_bootstrap_tps_hist")

            costructure_tps = np.array(n_tps(a_typed, costructure_ij))
            costructure_fps = np.array(n_fps(a_typed, costructure_ij))

            control_first_tps = np.array(n_tps(a_typed, control_ij))
            control_first_fps = np.array(n_fps(a_typed, control_ij))

            control_last_tps = np.array(n_tps(a_typed, control_last_ij))
            control_last_fps = np.array(n_fps(a_typed, control_last_ij))

            shuff_direct_tps = np.array(n_tps(a_typed, shuff_direct_ij))
            shuff_direct_fps = np.array(n_fps(a_typed, shuff_direct_ij))

            shuff_costructure_tps = np.array(n_tps(a_typed, shuff_costructure_ij))
            shuff_costructure_fps = np.array(n_fps(a_typed, shuff_costructure_ij))
            
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

            plot_plot(shuff_costructure_tps, score, "k.", "shuff costructure tp edges", "score", "shuff_costructure_tp_vs_score" + suffix) 
            plot_plot(shuff_costructure_fps, score, "k.", "shuff costructure fp edges", "score", "shuff_costructure_fp_vs_score" + suffix) 

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
                    shuff_costructure_tps = shuff_costructure_tps,
                    shuff_costructure_fps = shuff_costructure_fps,
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
                        "r_score", "sij_score", "z_score", "n_edges_score", "grad_r_score", "grad_z_score"):
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

        n_direct_ps = np.sum(direct_ij)
        n_costructure_ps = np.sum(costructure_ij)

        df = pd.DataFrame({"n_positives": [n_direct_ps, n_costructure_ps]}, index=["direct", "costructure"])
        df.to_csv(str(o / "summary.tsv"), sep="\t")
    
    def multi_chain_sampling_analysis(i=i, o=o):
        # Reset i and o to reference the merged directory
        logging.info("multi_chain_sampling_analysis")
        # Load the model data
        with open(str(i.parent) + "/merged_results_model_data.pkl", "rb") as f:
            model_data = pkl.load(f)
        i_ = i
        
        if not _dev:
            model_data_plots(model_data)

        direct_ref = data_io.get_pdb_ppi_predict_direct_reference()
        costructure_ref = data_io.get_pdb_ppi_predict_cocomplex_reference()


        direct_ij = align_reference_to_model(model_data, direct_ref, mode=mode) 
        costructure_ij = align_reference_to_model(model_data, costructure_ref, mode=mode)

        rng_key = jax.random.PRNGKey(122387)
        rng_key, key  = jax.random.split(rng_key)

        shuff_costructure_ij = np.array(jax.random.permutation(key, costructure_ij))

        rng_key, key  = jax.random.split(rng_key)
        shuff_direct_ij = np.array(jax.random.permutation(key, direct_ij))
        x : dict = postprocess_samples(i.parent, fbasename=i.stem, merge=True) 
        with open(str(i) + ".pkl", "rb") as f:
            x2 = pkl.load(f)
        
        N = model_data['N']
        
        fplot = partial(
            run_multichain_specific_plots,
            model_data = model_data,
            save=save, o = o)
        fplot(x, suffix="_w_warmup")
        fplot(x2)

        gplot = partial(
            run_multi_chain_version_of_single_chain_plots,
            model_data = model_data,
            save = save,
            o = o)
                
        gplot(x = x, suffix = "_w_warmup")
        gplot(x2)



    if not merge: 
        single_chain_sampling_analysis()
    else:
        multi_chain_sampling_analysis()
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
