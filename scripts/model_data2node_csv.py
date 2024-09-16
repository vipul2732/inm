import click
from pathlib import Path
import pickle as pkl
import pandas as pd

@click.command()
@click.option("--i")
@click.option("--o")
def main(i, o):
    if not isinstance(i, Path):
        i = Path(i)
    
    if not isinstance(o, Path):
        o = Path(o)
    
    assert o.is_dir()
    assert i.is_file()

    with open(i, "rb") as f:
        model_data = pkl.load(f)
    
    node_idx2name = model_data["node_idx2name"]
    N = len(node_idx2name)
    names = []
    for j in range(N):
        names.append(node_idx2name[j])
    assert len(names) == len(list(set(names)))

    df = pd.DataFrame({"#names": names})

    df.to_csv(o / "node_names.tsv", sep="\t", index = False)


def _main():
    main(i, o)


if __name__ == "__main__":
    main()
