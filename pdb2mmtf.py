"""
Given a pdbfile, convert it to an mmtf file
"""
from pathlib import Path
import click
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb


@click.command()
@click.option("-i", required=True, help="pdb file path")
@click.option("-o", required=True, help="output dir")
def main(i, o):
    pdb_file_path = i
    output_dir = o

    db_file = pdb.PDBFile.read(pdb_file_path)
    struct = db_file.get_structure()

    pdb_id = Path(pdb_file_path).name.removesuffix(".pdb")
    output_path = f"{output_dir}/{pdb_id}.mmtf"
    
    mmtf_file = mmtf.MMTFFile()
    mmtf.set_structure(mmtf_file, struct)
    mmtf_file.write(output_path)

if __name__ == "__main__":
    main()
