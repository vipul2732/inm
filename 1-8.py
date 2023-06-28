"""
Fetch PDB Files.

How to handle edge cases?
- Large structures cannot be represented by a single PDB file.
- Could you multiple PDB Files to represent a single structure
- Could use the mmcif
- Could use an MMTF
- How many edge cases are there?
"""
import pandas as pd
import sys
from pathlib import Path
import biotite.database.rcsb

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

df = pd.read_csv("hhblits_out/SignificantNonSelfPrey.csv")

pdb_ids = [i.split("_")[0] for i in df['PDB70ID1'].unique()]
pdb_dir = Path("pdb_files")

failed = []

for pdb_id in pdb_ids:
    if pdb_id not in [i.name for i in pdb_dir.iterdir()]:
        try:
            file_path = biotite.database.rcsb.fetch(pdb_id, "pdb", pdb_dir)
            print(file_path)
        except biotite.database.RequestError:
            eprint(f"Request Error {file_path}. Skipping.")
            failed.append(file_path)

failed_log = pd.DataFrame({"1-6Failed": failed})

failed_log.to_csv("1-6Failed_log.csv", index=False)
