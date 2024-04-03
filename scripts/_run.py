"""
0. Define inputs and parameters 
1. Preprocess inputs to an output dir
2. Choose the scoring function
3. Run sampling
4. Generate the sampling figures
5. Generate the sampling tables
6. Generate the benchmark figures
7. Generate the benchmark tables
"""
import click
import sys
from pathlib import Path
import logging

sys.path.append("../notebooks")

import preprocess_data
import _model_variations as mv
import generate_sampling_figures as gsf
import generate_benchmark_figures as gbf
import generate_analysis_figures as gaf
import json
import shutil

def preprocessor_protocol_dispatcher(preprocessor_protocol_name):
    if preprocessor_protocol_name == "cullin_standard":
        return preprocess_data.preprocess_spec_table
    else:
        raise NotImplementedError

def preprocess_model():
    ...

def conditionally_mkdir(d):
    if d.is_dir():
        ...
    else:
        d.mkdir()

def main(
    model_output_dirpath,    
    model_input_fpath,
    preprocessor_protocol_name,
    preprocessor_output_dirpath,
    model_id,
    rseed,
    model_name,
    model_data,
    num_warmup,
    num_samples,
    include_potential_energy,
    include_mean_accept_prob, 
    include_extra_fields, 
    progress_bar, 
    initial_position, 
    save_warmup, 
    load_warmup, 
    jax_profile,
    hyper_param_alpha,
    hyper_param_beta,
    hyper_param_thresholds,
    filter_kw,
):
    model_output_dirpath = Path(model_output_dirpath)
    model_input_fpath= Path(model_input_fpath)
    preprocessor_output_dirpath = Path(preprocessor_output_dirpath)
    
    conditionally_mkdir(model_output_dirpath)

    #setup_logging(model_output_dirpath)

    preprocess_inputs = preprocessor_protocol_dispatcher(preprocessor_protocol_name) 

    preprocess_inputs(
            input_path = model_input_fpath,
            output_dir = preprocessor_output_dirpath,
            sheet_nums = 3,
            prey_colname = "PreyGene",
            enforce_bait_remapping = True,
            filter_kw = filter_kw,
            )

    # Copy the files from the preprocessed data to the modeling output dir
    shutil.copy(preprocessor_output_dirpath / "spec_table.tsv", model_output_dirpath / "spec_table.tsv")
    shutil.copy(preprocessor_output_dirpath / "composite_table.tsv", model_output_dirpath / "composite_table.tsv")
    shutil.copy("../notebooks/shuffled_apms_correlation_matrix.pkl",
                model_output_dirpath / "shuffled_apms_correlation_matrix.pkl")

    modeler_vars = {"lower_edge_prob" : hyper_param_alpha,
                    "upper_edge_prob" : hyper_param_beta,
                    "thresholds" : hyper_param_thresholds} 

    with open(str(model_output_dirpath / "modeler_vars.json"), "w") as f:
        json.dump(modeler_vars, f)



    mv._main(model_id = model_id,
             rseed = rseed,
             model_name = model_name,
             model_data = model_data,
             num_warmup = num_warmup,
             num_samples = num_samples,
             include_potential_energy = include_potential_energy,
             include_mean_accept_prob = include_mean_accept_prob,
             include_extra_fields = include_extra_fields,
             progress_bar = progress_bar,
             save_dir = str(model_output_dirpath),
             initial_position = initial_position,
             save_warmup = save_warmup,
             load_warmup = load_warmup,
             jax_profile = jax_profile)
    
    input_file = model_output_dirpath / f"{model_id}_{model_name}_{rseed}.pkl"
    gsf._main(o = str(model_output_dirpath),
              i = input_file)

    gbf.cullin_standard()

    gaf.cullin_standard()

