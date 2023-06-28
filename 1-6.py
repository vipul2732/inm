"""
Fetch PDB Files.
"""
import pandas as pd
import sys
from pathlib import Path
import biotite.database.rcsb

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

df = pd.read_csv("hhblits_out/PreyPDB70PairAlign.csv")

pdb_ids = [i.split("_")[0] for i in df['PDB70ID'].unique()]
pdb_dir = Path("pdb_files")

for pdb_id in pdb_ids:
    if pdb_id not in [i.name for i in pdb_dir.iterdir()]:
        try:
            file_path = biotite.database.rcsb.fetch(pdb_id, "pdb", pdb_dir)
            print(file_path)
        except biotite.database.RequestError:
            eprint(f"Request Error {file_path}. Skipping.")


