import click
import _run
import run_configurations as rc
from pathlib import Path
import json


@click.command()
@click.argument("name") 
@click.option("--figures", default = False, is_flag = True, help="only run figures")
def main(name, figures):
    run_configuration =rc.__dict__[name]._asdict() 
    model_output_dirpath = Path(run_configuration['model_output_dirpath'])
    if not figures: 
        _run.main(**run_configuration)
        jsonpath = model_output_dirpath  / "run_config.json"
        with open(str(jsonpath), "w") as f:
            json.dump(run_configuration, f)
    else:
        _run.figures(**run_configuration)

if __name__ == "__main__":
    main()
