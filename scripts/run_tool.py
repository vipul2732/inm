import click
import _run
import run_configurations as rc
from pathlib import Path
import json


@click.command()
@click.argument("name") 
@click.option("--figures", default = False, is_flag = True, help="only run figures")
@click.option("--rseed", default = None)
def main(name, figures, rseed):
    run_configuration =rc.__dict__[name]._asdict() 
    if rseed is not None:
        rseed = int(rseed)
        # update the run configuration dynamically
        update_rc_dict_rseed(run_configuration, rseed)
    model_output_dirpath = Path(run_configuration['model_output_dirpath'])
    if not figures: 
        _run.main(**run_configuration)
        jsonpath = model_output_dirpath  / "run_config.json"
        with open(str(jsonpath), "w") as f:
            json.dump(run_configuration, f)
    else:
        _run.figures(**run_configuration)

def update_rc_dict_rseed(rc_dict, rseed):
    rc_dict["rseed"] = rseed
    rc_dict["model_output_dirpath"] = rc_dict["model_output_dirpath"] + f"_rseed_{rseed}"

if __name__ == "__main__":
    main()
