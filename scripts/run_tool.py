import json
import pickle as pkl
import logging
import sys

import click
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from pathlib import Path

import run_configurations as rc
import _run
sys.path.append("../notebooks")
from sampling_assessment import (
        get_results
)


@click.command()
@click.argument("name") 
@click.option("--figures", default = False, is_flag = True, help="only run figures")
@click.option("--rseed", default = None)
@click.option("--merge", is_flag = True, default = False, help="merge multiple chains") 
def main(name, figures, rseed, merge):
    if not merge:
        run_configuration =rc.__dict__[name]._asdict() 
        if rseed is not None:
            rseed = int(rseed)
            # update the run configuration dynamically
            run_configuration = update_rc_dict_rseed(run_configuration, rseed)
        model_output_dirpath = Path(run_configuration['model_output_dirpath'])
        if not figures: 
            _run.main(**run_configuration)
            jsonpath = model_output_dirpath  / "run_config.json"
            with open(str(jsonpath), "w") as f:
                json.dump(run_configuration, f)
        else:
            _run.figures(**run_configuration)
    else:
        do_merge(name)

class MergeObject:
    def __init__(self, results = None, model_data = None, hmc_warmup = None):
        self.results = results
        self.model_data = model_data
        self.hmc_warmup = hmc_warmup

def do_merge(name):
    """ Merge multiple input chains into a single output directory """
    out = Path(f"../results/{name}_merged")
    if not out.is_dir():
        out.mkdir()
    # set up logging
    
    logging.basicConfig(
        filename = str(out / "_run.log"),
        level=logging.DEBUG,
        filemode="w",
        format = "%(levelname)s:%(module)s:%(funcName)s:%(lineno)s:MSG:%(message)s",
    ) 
    logging.info(f"merging {name} chains")
    base = Path(f"../results/{name}")
    logging.info(f"base: {base}")
    #print("base", base)

    assert "rseed" not in base.name, "merge should exclude '_rseed_x'"
    
    # make the output directory or overwrite
    if not out.is_dir():
        out.mkdir()
    
    merge_object = MergeObject(results=None, model_data=None) 
    merge_object = update_merge_object(name, merge_object, base)
    write_merged_results(merge_object, out)

def update_merge_object(name, merge_object, base):
    for path in base.parent.iterdir():
        if (path.is_dir()) and ("rseed" in path.name) and ("merge" not in path.name) and (name in path.name):
            merge_object = update_merge_object_from_chain_path(name, merge_object, path)
    return merge_object

def update_merge_object_from_chain_path(name, merge_object, chain_path):
    merge_object = update_merge_object_warmup_hmc(merge_object, chain_path)
    merge_object = update_merge_object_results(name, merge_object, chain_path)
    merge_object = update_merge_object_model_data(merge_object, chain_path)
    return merge_object

def find_hmc_warmup_file_path(chain_path):
    for path in chain_path.iterdir():
        if path.is_file() and ("hmc_warmup" in path.name):
            return path

def find_results_file_path(name, chain_path):
    pkl_files = [path for path in chain_path.iterdir() if (path.is_file()) and (".pkl" == path.suffix)]
    pkl_files = [path for path in pkl_files if name in path.name] 
    results_file = [path for path in pkl_files if (("hmc_warmup" not in path.name) and ("model_data" not in path.name))] 
    assert len(results_file) == 1, f"results file not found in {chain_path}"
    results_file_path = results_file[0]
    return results_file_path

def pklload(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def dictionary_expand_dims(dictionary) -> dict:
    return tree_map(lambda x: jnp.expand_dims(x, axis=0), dictionary)

def dictionary_concat_chains(x: dict, y: dict) -> dict:
    return tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), x, y)

def update_merge_object_warmup_hmc(merge_object, chain_path):
    warmup_hmc_path = find_hmc_warmup_file_path(chain_path) 
    print(warmup_hmc_path)
    hmc_warmup = dictionary_expand_dims(pklload(warmup_hmc_path))
    if merge_object.hmc_warmup is None:
        merge_object.hmc_warmup = hmc_warmup 
    else:
        merge_object.hmc_warmup = dictionary_concat_chains(merge_object.hmc_warmup, hmc_warmup)
    return merge_object

def update_merge_object_results(name, merge_object, chain_path):
    results_path = find_results_file_path(name, chain_path)
    results = dictionary_expand_dims(pklload(results_path))
    if merge_object.results is None:
        merge_object.results = results
    else:
        merge_object.results = dictionary_concat_chains(merge_object.results, results)
    return merge_object

def update_rc_dict_rseed(rc_dict, rseed):
    rc_dict["rseed"] = rseed
    rc_dict["model_output_dirpath"] = rc_dict["model_output_dirpath"] + f"_rseed_{rseed}"
    return rc_dict

if __name__ == "__main__":
    main()
