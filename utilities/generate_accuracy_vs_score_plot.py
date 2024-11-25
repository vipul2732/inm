import click
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append("../notebooks")

@click.command()
@click.option("--dpath", help="path to the output directory")
def main(dpath):
    ...
    
def _main(dpath):
    # Read in the models w/ warmup

    # Read in the scores

    # Add supplementary models and scores to the plot

    # plot the figure  

    fname = "dummy"
    plot_figure()

def plot_figure():

    plt.xlabel("score")
    plt.ylabel("accuracy")
    plt.savefig(dpath / (fname + "_300.png"), dpi=300)
    plt.savefig(dpath / (fname + "_1200.png"), dpi=1200)


if __name__ == "__main__":
    main()
