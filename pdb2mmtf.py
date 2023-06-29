"""
Given a pdbfile, convert it to an mmtf file
"""
from pathlib import Path
import click
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx


@click.command()
@click.option("-i", required=True, help="pdb file path")
@click.option("-o", required=True, help="output dir")
@click.option("-from", "from_", default="pdb")
@click.option("-to", default="mmtf")
def main(i, o, from_, to):
    if from_ == "pdb": 
        Reader = pdb.PDBFile
        read_suffix = ".pdb"
        read_mod = pdb
    elif from_ == "cif":
        Reader = pdbx.PDBxFile
        read_suffix = ".cif"
        read_mod = pdbx

    if to == "mmtf":
        write_mod = mmtf
        write_suffix = "mmtf"
        Writer = mmtf.MMTFFile


    pdb_file_path = i
    output_dir = o

    db_file = Reader.read(pdb_file_path)
    stack = read_mod.get_structure(db_file)
    assert len(stack) == 1, f"multiple models in {pdb_file_path}"
    array = stack[0]

    pdb_id = Path(pdb_file_path).name.removesuffix(read_suffix)
    output_path = f"{output_dir}/{pdb_id}.{write_suffix}"
    
    write_file = Writer()
    write_mod.set_structure(write_file, array)
    write_file.write(output_path)

if __name__ == "__main__":
    main()
