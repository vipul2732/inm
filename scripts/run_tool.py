import click
import _run
import run_configurations as rc
from pathlib import Path
import json

@click.command()
@click.argument("name")
def main(name):
    run_configuration =rc.__dict__[name]._asdict() 
    jsonpath = Path(run_configuration['model_output_dirpath'] / "run_config.json")
    with open(str(jsonpath), "w") as f:
        json.dump(run_configuration, f)
    _run.main(**run_configuration)

if __name__ == "__main__":
    main()
