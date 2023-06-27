import subprocess
import pandas as pd

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

fastas = [i for i in Path("input_sequences").iterdir() if i.suffix == ".fasta"]

for fasta_file in fastas:
    uid = fasta_file.removesuffix(".fasta") 
    hhblits_cmd = f"hhblits -i input_sequences/{uid}.fasta -o hhblits_output/{uid}.hhr"

