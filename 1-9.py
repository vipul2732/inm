"""
1. Load in the SignificantNonself Pair Prey File
2. For each pdb file
   2a. Load in the query sequences
   2b. For each query sequences
       2aa. Load in the sequence-chain-id pairs for the pdb file
       2ac. Filter to polypeptide sequecnes > 88 amino acids long
       2ad. Do a pairwise sequence alignment to every sequence in the pdb file
       2ae. If the alignment is > 88 aa, evalue < 1e-7, and 30% sequence id then add the sequence to the remap hit table
"""

import pandas as pd
from pathlib import Path
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf
import biotite.sequence.align

sig_alns = pd.read_csv("hhblits_out/SignificantNonSelfPairs.csv")
expected_pdbs = set([i.split("_")[0].lower() for i in sig_alns['PDB701'].values])
actual_pdbs = set([i.name.removesuffix(".mmtf") for i in Path("significant_cifs").iterdir() if i.suffix == ".mmtf"])
diff = expected_pdbs.symmetric_difference(actual_pdbs)
#assert len(diff) == 0, (len(diff), diff, actual_pdbs, expected_pdbs)

remapping_columns = ["QueryID", "ChainID", "PDBID", "bt_aln_cols", "bt_evalue", "bt_psid", "bt_Q", "bt_T", "Qseq"]

QueryID = []
ChainID = []
PDBID = []
bt_aln_cols = []
bt_evalue = []
bt_psid = []
bt_Q = []
bt_T = []
Qseq = []

df = sig_alns

blossum62 = biotite.sequence.align.SubstitutionMatrix.std_protein_matrix()
uids = set(df["Query1"].values).union(df["Query2"].values)
sequences = {}

ProtSeq = biotite.sequence.ProteinSequence

res_id_map = {"ALA":"A", "ASP": "D", "CYS": "C", "GLY": "G", "HIS": "H"} 

def get_sequence(atom_array, chain_id):
    mask = atom_array.chain_id == chain_id
    resids = atom_array.res_id[mask]
    res_names = atom_array.res_name[mask]
    Natoms = len(resids)
    N = max(resids) - min(resids)
    seq = ["X"] * N 
    mapp = {resids[i]: ProtSeq.convert_letter_3to1(res_names[i]) for i in range(Natoms)}
    for i in range(N): 
        if i in mapp:
            seq[i] = mapp[i]
    return "".join(seq) 

def get_sequence_dict(atom_array):
    chain_ids = set(atom_array.chain_id)
    return {chain_id: get_sequence(atom_array, chain_id) for chain_id in chain_ids}

for uid in uids:
    fasta_file = biotite.sequence.io.fasta.FastaFile.read(f"input_sequences/{uid}.fasta")
    assert len(fasta_file) == 1
    header, sequence = list(fasta_file.items())[0]
    sequences[uid]=sequence
    
bio_assemblies = {}
bio_sequences = {}
for pdb_id in expected_pdbs:
    read_f = biotite.structure.io.mmtf.MMTFFile.read(f"significant_cifs/{pdb_id}.bio.mmtf")
    bio_array = biotite.structure.io.mmtf.get_structure(read_f, model=1)
    #bio_assemblies[pdb_id] = bio_array
    mask = bio_array.hetero == False  # Exclude heteroatoms
    bio_array = bio_array[mask]
    seq_dict = get_sequence_dict(bio_array)
    bio_sequences[pdb_id] = seq_dict
    
for pdb_id in expected_pdbs:
    sel = [pdb_id == i.lower()[0:4] for i in sig_alns['PDB701'].values] 
    subframe = df[sel]

    queries = set(subframe["Query1"].values).union(subframe["Query2"].values)
    
    stack = biotite.structure.io.mmtf.MMTFFile.read(f"significant_cifs/{pdb_id}.bio.mmtf")
    assert len(stack) == 0
    array = stack[0]
    for query_id in queries: 
        fasta_file = biotite.sequence.io.fasta.FastaFile.read(f"input_sequences/{query_id}.fasta")
        assert len(fasta_file) == 1
        header, sequence = list(fasta_file.items())[0]

    break

out_df = pd.DataFrame({"QueryID": QueryID, "ChainID": ChainID,
        "PDBID": PDBID, "bt_aln_cols": bt_aln_cols, "bt_evalue": bt_evalue,
        "bt_psid": bt_psid, "bt_Q": bt_Q, "bt_T": bt_T, "Qseq": Qseq})

print(out_df)
