import pandas as pd
from pathlib import Path
import click

@click.command()
@click.option("--path")
def main(path):
    path = Path(path)
    sc = pd.read_csv(str(path / "spec_table.tsv"), sep="\t")
    c = pd.read_csv(str(path / "composite_table.tsv"), sep="\t")
    print(f"INFO::summarize_processed_data::{path}/spec_table.tsv: {sc.shape}") 
    print(f"INFO::summarize_processed_data::{path}/composite_table.tsv: {c.shape}") 

if __name__ == "__main__":
    main()
