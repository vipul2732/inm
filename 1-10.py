"""
1. Read in the BSASA DataFrame
2. Remove entries with BSASA less than 500 square angstroms BSASA

3. Compute a PDBID -> ChainID -> Preys mapping
4. Compute the prey x prey direct interaction positives 
5. Compute the prey x prey co-complex  
6. Compute the prey x prey indirect interaction (co-complex & < 500 square angstroms) 
"""

import pandas as pd
from pathlib import Path
from itertools import combinations
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.align
import sys
import numpy as np

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

hhblits_df = pd.read_csv("hhblits_out/SignificantPreyPDB70PairAlign.csv")
hhblits_cols = hhblits_df.columns
chain_mapping = pd.read_csv("significant_cifs/chain_mapping.csv")

bsasa = pd.read_csv("significant_cifs/BSASA_concat.csv")
eprint(f"N bsasa {len(bsasa)}")

# Drop the key errors
sel = bsasa["BSASA"] != "KeyError"
bsasa = bsasa[sel]

bsasa["BSASA"] = np.array([float(i) for i in bsasa["BSASA"].values])

assert bsasa["BSASA"].values.dtype == np.float64
sel = bsasa["BSASA"].values >= 500
bsasa = bsasa[sel]
eprint(f"N bsasa {len(bsasa)} 500")
seqid_col = "bt_aln_percent_seq_id"
chain_mapping.loc[:, seqid_col] = np.array([float(i) for i in chain_mapping[seqid_col].values])
sel = chain_mapping[seqid_col] >= 0.3
chain_mapping = chain_mapping[sel]
eprint(f"N mapped chains {sum(sel)}")

uid_set = sorted(list(set(chain_mapping["QueryID"].values)))
pdb_set = sorted(list(set(bsasa["PDBID"].values)))
uid_pairs = list(combinations(uid_set, 2))

#dims = ["preyu", "preyv", "pdb_id"]
#coords = {"preyu": uid_set, "preyv": uid_set, "pdb_id": pdb_set}

# PDBID -> ChainID -> Prey-IDs 

# (prey1, prey2) -> pdb_id -> (chain1, chain2) -> BSASA

def fetch_prey(chain_mapping_subframe, pdb_id, chain_id):
    sel = chain_mapping_subframe["ChainID"] == chain_id
    chain_df = chain_mapping_subframe[sel]
    result = ""
    for i, r in chain_df.iterrows():
        prey = r["QueryID"]
        result = result + ";" + prey
    return result


prey1_lst = []
prey2_lst = []
pdb_id_lst = []
chain1_lst = []
chain2_lst = []
bsasa_lst = []

bsasa_pdb_set = sorted(set(bsasa["PDBID"].values))
for pdb_id in bsasa_pdb_set:
    chain_subframe = chain_mapping[chain_mapping["PDBID"] == pdb_id]
    bsasa_subframe = bsasa[bsasa["PDBID"] == pdb_id]
    assert len(bsasa_subframe) > 0, pdb_id

    bsasa_chain_set = set(bsasa_subframe["Chain1"].values).union(bsasa_subframe["Chain2"].values)
    chain_pairs = list(combinations(bsasa_chain_set, 2))

    chain2prey_lst = {chain: fetch_prey(chain_subframe, pdb_id, chain) for chain in bsasa_chain_set}


    chainpair2bsasa = {}
    for i, r in bsasa_subframe.iterrows():
        chain1 = r["Chain1"]
        chain2 = r["Chain2"]
        #assert isinstance(chain1, str), (chain1, pdb_id)
        #assert isinstance(chain2, str), (chain2, pdb_id)
        bsasa_calc = r["BSASA"]
        chainpair2bsasa[frozenset((chain1, chain2))] = bsasa_calc

    for (chain1, chain2) in chain_pairs:
        #assert isinstance(chain1, str), (chain1, pdb_id)
        #assert isinstance(chain2, str), (chain2, pdb_id)

        chain_key = frozenset((chain1, chain2))

        if chain_key not in chainpair2bsasa:
            continue
        preylst_1 = chain2prey_lst[chain1]
        preylst_2 = chain2prey_lst[chain2]

        if (len(preylst_1) == 0) or (len(preylst_2) == 0):
            pass
        else:
            preylst_1 = preylst_1.split(";")
            preylst_2 = preylst_2.split(";")
            
            preyset_1 = set(preylst_1)
            preyset_2 = set(preylst_2)

            for prey1 in preyset_1:
                for prey2 in preyset_2:
                    prey1_lst.append(prey1)
                    prey2_lst.append(prey2)
                    pdb_id_lst.append(pdb_id)
                    chain1_lst.append(chain1)
                    chain2_lst.append(chain2)
                    bsasa_lst.append(chainpair2bsasa[chain_key])

df = pd.DataFrame({"Prey1": prey1_lst,
  "Prey2": prey2_lst,
  "PDBID": pdb_id_lst,
  "Chain1": chain1_lst,
  "Chain2": chain2_lst,
  "bsasa_lst": bsasa_lst})

df.to_csv("significant_cifs/BSASA_reference.csv", index=False)
