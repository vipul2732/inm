import click
import _run
import run_configurations as rc
from pathlib import Path
import json

import sys
sys.path.append("../notebooks")
from sampling_assessment import (
        get_results
)


@click.command()
@click.argument("name") 
@click.option("--figures", default = False, is_flag = True, help="only run figures")
@click.option("--rseed", default = None)
@click.option("--merge", type=str, default = None)
def main(name, figures, rseed, merge):
    if merge is None:
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
        do_merge(merge)

def do_merge(merge : str):
    """ Merge multiple input chains into a single output directory """
    out = Path(merge + "_merged")
    base = Path(merge)

    assert "rseed" not in base.name, "merge should exclude '_rseed_x'"
    
    # make the output directory or overwrite
    if not out.is_dir():
        out.mkdir()
    
    merge_object = None 
    for fpath_or_dpath in base.parent.iterdir():
        if fpath_or_dpath.is_file():
            continue
        if base.name in fpath_or_dpath.name:
            merge_object = update_merge_object(merge_object, fpath_or_dpath)
    
    write_merge_object_to_dir(merge_object, out)

def init_merge_object():
    """ The merge object contains
    """
    ...


def update_merge_object(merge_object, dpath):
    rseed = int(str(dpath).split("_rseed_")[-1]) 
    pkl_files = [x.name for x in dpath.iterdir() if x.suffix == ".pkl"]

    match_str = "_hmc_warmup.pkl"
    for x in pkl_files:
        if match_str in x:
            mname = x.split(match_str)[0]
            break
    results = get_results(dpath, mname=mname, rseed=rseed)
    if merge_object is None:
        return init_merge_object(results)
    else:
        return stack_along(merge_object, results)

def init_merge_object(results):
    """ Add a leading axis to the important arrays """

def stack_along(merge_object, results):
    """
    Stack chains along the 0th axis for the following fields
    """
    merge_keys = ("samples", "extra_fields")
    out = {}

    for merge_key in merge_keys:
        merge_val = merge_object[key]
        results_val = results[key]
        out[merge_key] = {}
        for sub_key in merge_val.keys():
            out[merge_key][sub_key] = np.stack([a, b])
            

def write_merge_object_to_dir(merge_object, out):
    ...

def update_rc_dict_rseed(rc_dict, rseed):
    rc_dict["rseed"] = rseed
    rc_dict["model_output_dirpath"] = rc_dict["model_output_dirpath"] + f"_rseed_{rseed}"
    return rc_dict

if __name__ == "__main__":
    main()
