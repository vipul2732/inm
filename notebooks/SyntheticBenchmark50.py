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

# # SyntheticBenchmark50
# Do a synthetic benchmark with a 50 node network and model14
# ## -1
# 1. Imports
# 2. Make a directory for the outputs of this notebook if it doesn't exist
# 3. Set up logging
# 4. Global variables
# ## 0 Ground Truth Generation
# 1. Choose 3 bait and 50 node
# 2. Generate a network of direct protein-protein interactions
# 3. Ensure everything is connected to the bait
# ## 1 Data Generation
# 1. Generate profile similarities given the network
# ## 2 Fitting
# 1. Generate M chains of N samples each, saving to files
# 2. Ensure the score is saved
# 3. Optionally check sampling diagnostics
# ## 3 Analysis
# 1. Calculate accuracy for every position - (True accuracy metric)
# 2. Save the top accuracy
# 3. Save the top scoring model
# 2. Accuracy score precision
# ## 4 Increased Sampling
# 1. Calculate a dictionary of increasing combinations of the M chains
# ## 5 Plots
# 1. Plot accuracy score correlation for a representative trajectory
# 2. Plot top accuracy top score calculation across all trajectories
# 3. Show top accuracy as a function of increased sampling
# 4. Show top score as a function of increased sampling
# 5. Finish

# +
## -1.1 Imports
import datetime
import importlib
import jax
import jax.numpy as jnp
import logging
import networkit as nk
import numpy as np
import math
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
# Custom modules
import synthetic_benchmark
import _model_variations
import analyze_mcmc_output

# Reload custom modules for devlopment
importlib.reload(synthetic_benchmark)
importlib.reload(_model_variations)
importlib.reload(analyze_mcmc_output)

# -

## -1.2 Make a directory if it doesn't exist
nb_write_path = Path("SyntheticBenchmark50")
if not nb_write_path.is_dir():
    nb_write_path.mkdir()

## -1.3
# Start a new log file to write to
time = datetime.datetime.now()
log_filename = nb_write_path / f"{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}_SyntheticBenchmark50.log"
# Set up the logger
logging.basicConfig(filename=str(log_filename), encoding="utf-8", level=logging.DEBUG)
logging.info(f"SyntheticBenchmark50 Notebook Begin at {str(time)}")

## -1.4
nb = SimpleNamespace()
nb.rseed = 13
nb.rkey = jax.random.PRNGKey(nb.rseed)
nb.n_keys = 3
nb.keys = jax.random.split(nb.rkey, nb.n_keys)
nb.dev_key = jax.random.PRNGKey(22)
nb.n_prey = 50
nb.tril_idx = np.tril_indices(nb.n_prey)
nb.num_samples = 1000
nb.m_chains = 1000
nb.num_warmup = 500


# +
## 0.1 Ground Truth Generation
def _0_1_ground_truth_generation():
    A, bait_idx, prey_idx = synthetic_benchmark.get_bait_prey_network(nb.keys[0], n_prey=50,
                                                                     n_bait=3, d_crit=20)
    return A, bait_idx, prey_idx

A, bait_idx, prey_idx = _0_1_ground_truth_generation()
# -

bait_idx

A.plot()
plt.hlines(np.array(bait_idx), -1, 50, label="Bait", alpha=0.8, color='r')
plt.legend()

D = _model_variations.shortest_paths_up_to_23(A.values)

plt.matshow(D)
plt.colorbar(shrink=0.8)


# +
def adjacency2graph(A, weighted=False):
    n, m = A.shape
    G = nk.Graph(n, weighted=weighted)
    for u in range(n):
        for v in range(u+1, m):
            w = A[u, v]
            if weighted:
                G.addEdge(u, v, w)
            else:
                if w > 0:
                    G.addEdge(u, v, w)
    return G
                
            
# -

G = adjacency2graph(A)

nk.plot.connectedComponentsSizes(G)
plt.title("connected component sizes") # A single connected component

# 1.1 Generate similarities
data = synthetic_benchmark.data_from_network_model14_rng(nb.keys[1], A.values)

# +

plt.matshow(data)
plt.colorbar(shrink=0.8)
# -

plt.matshow(A.values)
plt.colorbar(shrink=0.8)

data_flat =  _model_variations.matrix2flat(data)
assert len(data_flat) == math.comb(nb.n_prey, 2)

plt.hist(np.array(model_data['flattened_apms_similarity_scores']), bins=20)
plt.show()


# +
## 2.1 Fitting
def _2_1_fit():
    for i in range(nb.m_chains):
        savename = Path("SyntheticBenchmark50/") / f"chain_{i}.pkl"
        if not Path(savename).is_file():
            logging.info(f"BEGIN sampling {savename}")
            _model_variations._main(
            model_id="0",
            model_name="model14",
            rseed=i,
            model_data=data_flat,
            num_warmup=nb.num_warmup,
            num_samples=nb.num_samples,
            include_potential_energy=True,
            progress_bar=True,
            save_dir="SyntheticBenchmark50",
            include_mean_accept_prob=False,
            include_extra_fields=True)
            logging.info(f"END sampling {savename}")
        else:
            logging.info(f"SKIP sampling {savename} exists.")

_2_1_fit()


# -

## 3.1 Sampling Analysis
def _3_1():
    for i in range(nb.m_chains):
        # Load in the trajectory
        path = Path("SyntheticBenchmark50/") / f"chain_{i}.pkl"
        
        # do the sampling analysis
          # calc top score
          # calc accuracy array
          # get the top accuracy
          
          # 0_model14_0 : {top_score, accuracy_array, top_accuracy, potential_energy}
        


analyze_mcmc_output
