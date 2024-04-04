from typing import NamedTuple, Any, List

class RunConfiguration(NamedTuple):
    model_output_dirpath : str 
    model_input_fpath : str 
    preprocessor_protocol_name : str 
    preprocessor_output_dirpath : str
    model_id : int 
    rseed : int 
    model_name : int 
    model_data : Any 
    num_warmup : int
    num_samples : int 
    include_potential_energy : bool 
    include_mean_accept_prob : bool 
    include_extra_fields : bool
    progress_bar : bool 
    initial_position : Any
    save_warmup : bool 
    load_warmup : bool 
    jax_profile : bool 
    hyper_param_alpha : float 
    hyper_param_beta : float 
    hyper_param_thresholds : List[float]
    hyper_param_n_null_bins : int 
    hyper_param_disconectivity_distance : int 
    hyper_param_max_distance : int 
    filter_kw : str 

mini_dev_run = RunConfiguration(
   model_output_dirpath = "../results/mini_dev_run",
   model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
   preprocessor_protocol_name = "cullin_standard",
   preprocessor_output_dirpath = "../data/processed/cullin/",
   model_id = 0,
   rseed = 13,
   model_name = "model23_se",
   model_data = None,
   num_warmup = 10,
   num_samples = 10,
   include_potential_energy = True,
   include_mean_accept_prob = False,
   include_extra_fields = True,
   progress_bar = True,
   initial_position = None,
   save_warmup = True,
   load_warmup = True,
   jax_profile = False,
   hyper_param_alpha = 0.02,
   hyper_param_beta = 0.1,
   hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
   hyper_param_n_null_bins = 80,
   hyper_param_disconectivity_distance = 10,
   hyper_param_max_distance = 11,
   filter_kw = "all",
)

mini_se_sr_run = RunConfiguration(
    model_output_dirpath = "../results/mini_se_sr_run",
    model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = "../data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sr",
    model_data = None,
    num_warmup = 10,
    num_samples = 10,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 'auto',
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)

mini_se_sc_run = RunConfiguration(
    model_output_dirpath = "../results/mini_se_sc_run",
    model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = "../data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sc",
    model_data = None,
    num_warmup = 10,
    num_samples = 10,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 80,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)

se_all_10k =  RunConfiguration(
   model_output_dirpath = "../results/se_all_10k",
   model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
   preprocessor_protocol_name = "cullin_standard",
   preprocessor_output_dirpath = "../data/processed/cullin/",
   model_id = 0,
   rseed = 13,
   model_name = "model23_se",
   model_data = None,
   num_warmup = 1_000,
   num_samples = 10_000,
   include_potential_energy = True,
   include_mean_accept_prob = False,
   include_extra_fields = True,
   progress_bar = True,
   initial_position = None,
   save_warmup = True,
   load_warmup = True,
   jax_profile = False,
   hyper_param_alpha = 0.02,
   hyper_param_beta = 0.1,
   hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
   hyper_param_n_null_bins = 80,
   hyper_param_disconectivity_distance = 10,
   hyper_param_max_distance = 11,
   filter_kw = "all",
)


se_sr_all_10k =  RunConfiguration(
   model_output_dirpath = "../results/se_sr_all_10k",
   model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
   preprocessor_protocol_name = "cullin_standard",
   preprocessor_output_dirpath = "../data/processed/cullin/",
   model_id = 0,
   rseed = 13,
   model_name = "model23_se_sr",
   model_data = None,
   num_warmup = 1_000,
   num_samples = 10_000,
   include_potential_energy = True,
   include_mean_accept_prob = False,
   include_extra_fields = True,
   progress_bar = True,
   initial_position = None,
   save_warmup = True,
   load_warmup = True,
   jax_profile = False,
   hyper_param_alpha = 0.02,
   hyper_param_beta = 0.1,
   hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
   hyper_param_n_null_bins = 100,
   hyper_param_disconectivity_distance = 10,
   hyper_param_max_distance = 11,
   filter_kw = "all",
)

se_sc_all_10k =  RunConfiguration(
   model_output_dirpath = "../results/se_sc_all_10k",
   model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
   preprocessor_protocol_name = "cullin_standard",
   preprocessor_output_dirpath = "../data/processed/cullin/",
   model_id = 0,
   rseed = 13,
   model_name = "model23_se_sc",
   model_data = None,
   num_warmup = 1_000,
   num_samples = 10_000,
   include_potential_energy = True,
   include_mean_accept_prob = False,
   include_extra_fields = True,
   progress_bar = True,
   initial_position = None,
   save_warmup = True,
   load_warmup = True,
   jax_profile = False,
   hyper_param_alpha = 0.02,
   hyper_param_beta = 0.1,
   hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
   hyper_param_n_null_bins = 100,
   hyper_param_disconectivity_distance = 10,
   hyper_param_max_distance = 11,
   filter_kw = "all",
)

se_sr_sc_all_10k =  RunConfiguration(
   model_output_dirpath = "../results/se_sr_sc_all_10k",
   model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
   preprocessor_protocol_name = "cullin_standard",
   preprocessor_output_dirpath = "../data/processed/cullin/",
   model_id = 0,
   rseed = 13,
   model_name = "model23_se_sr_sc",
   model_data = None,
   num_warmup = 1_000,
   num_samples = 10_000,
   include_potential_energy = True,
   include_mean_accept_prob = False,
   include_extra_fields = True,
   progress_bar = True,
   initial_position = None,
   save_warmup = True,
   load_warmup = True,
   jax_profile = False,
   hyper_param_alpha = 0.02,
   hyper_param_beta = 0.1,
   hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
   hyper_param_n_null_bins = 100,
   hyper_param_disconectivity_distance = 10,
   hyper_param_max_distance = 11,
   filter_kw = "all",
)

