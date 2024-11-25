import pandas as pd
from pathlib import Path
import numpy as np 
import click


@click.command()
@click.option("--i")
@click.option("--o")
@click.option("--sf", default=3)
def main(i, o, sf):
    _main( i = i, o = o, sf=sf)

def _main(i, o, sf):
    i = Path(i)
    o = Path(o)
    df = pd.read_csv(i, sep="\t")
    w = [round(k, sf) for k in df['w']]
    df['w'] = w
    df.to_csv(o / (i.name.removesuffix(".tsv") + "_view.tsv"), index=False)


if __name__ == "__main__":
    main()
