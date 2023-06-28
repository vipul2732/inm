from pathlib import Path
import sys
sys.path = ["./src"] + sys.path
import click
import subprocess


@click.command()
@click.option('--start', default=0, help="start index inclusive")
@click.option('--end', default=1, help="end index exclusive")
@click.option('--db-path', default="~/databases/pdb70/")
def main(start, end, db_path):

    fastas = [i for i in Path("input_sequences").iterdir() if i.suffix == ".fasta"]
    
    for fasta_file in fastas[start:end]:
        uid = fasta_file.name.removesuffix(".fasta") 
        outf = f"hhblits_out/{uid}.hhr"
        hhblits_cmd = f"hhblits -i input_sequences/{uid}.fasta -o {outf}  -d {db_path}/pdb70" 
        hhr_files = [i.name for i in Path("hhblits_out").iterdir() if i.suffix == ".hhr"]
        
        if f"{uid}.hhr" not in hhr_files:    
            subprocess.call(hhblits_cmd, shell=True, stdout=open(outf, 'w'))

if __name__ == "__main__":
    main()
