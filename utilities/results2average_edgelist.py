"""
Given the results directory of a modeling run,
extract the average edge values and write them to a csv file.
"""
import click
from pathlib import Path
import pickle as pkl
import pandas as pd
import sys
sys.path.append("../notebooks")

import generate_sampling_figures as gsf

@click.command()
@click.option("--i")
@click.option("--o")
def main(i, o):
    if not isinstance(i, Path):
        i = Path(i)
    
    if not isinstance(o, Path):
        o = Path(o)
    
    assert not o.is_dir(), "output must be a file"
    assert i.is_dir()

    with open(i, "rb") as f:
        model_data = pkl.load(f)
    


def _main():
    main(i, o)


if __name__ == "__main__":
    main()
