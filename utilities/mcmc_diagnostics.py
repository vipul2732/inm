import click
import numpyro.diagnostics as diagnostics
import pickle as pkl
from pathlib import Path

@click.command()
@click.option("--i", help="model_output_dirpath")
@click.option("--fname")
def main(i, fname):
    _main(i, fname)

def _main(i, fname):
    i = Path(i)
    assert i.is_dir()
    samples_path = i / (fname + ".pkl")
    with open(samples_path, "rb") as f:
        results = pkl.load(f)
    samples = results['samples'] 
    sample_keys = list(samples.keys())
    k = 0
    for key, value in samples.items():
        print(key)
        print(value.shape)
        print(diagnostics.autocorrelation(value))
        #print(diagnostics.split_gelman_rubin(value))
        if k > 10:
            break


if __name__ == "__main__":
    main()
