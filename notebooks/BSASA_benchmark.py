# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# A reference set derived from the Protein Databank was obtained
#
# ## Conclusions
# - Saint recovers all 9 known direct bait prey interactions and all 19 known known bait-prey co-complex interactions

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from itertools import combinations
import math
import operator
from pathlib import Path
import numpyro
import numpyro.distributions as dist
import biotite.sequence.io
import jax.numpy as jnp
import jax
from numpyro.diagnostics import hpdi, summary


# +
# Some sanity checks
def h(x):
    return '{:,}'.format(x)

def preysel(df, prey_name):
    uid = gene2uid[prey_name]
    sel1 = df['Prey1'] == uid
    sel2 = df['Prey2'] == uid
    return df[sel1 | sel2]

def parse_spec(df, spec_colname='Spec', ctrl_colname='ctrlCounts',
              n_spec=4, n_ctrl=12):
    """
    Parse the Spec and CtrlCounts columns
    """
    
    rsel = [f"r{i}" for i in range(1, n_spec + 1)]
    csel = [f"c{i}" for i in range(1, n_ctrl + 1)]
    
    specs = np.array([list(map(int, i.split("|"))) for i in df[spec_colname].values])
    ctrls = np.array([list(map(int, i.split("|"))) for i in df[ctrl_colname].values])
    
    N = len(df)
    
    assert specs.shape == (N, n_spec)
    assert ctrls.shape == (N, n_ctrl)
    
    for i, rcol in enumerate(rsel):
        df.loc[:, rcol] = specs[:, i]
    
    for i, ccol in enumerate(csel):
        df.loc[:, ccol] = ctrls[:, i]
        
    return df


# -

bsasa_ref = pd.read_csv("../significant_cifs/BSASA_reference.csv")

# +
uid2seq = {}
tmp = [i for i in Path("../input_sequences/").iterdir() if ".fasta" in str(i)]
#assert len(tmp) == 3062, len(tmp)
for i in tmp:
    uid = i.name.removesuffix(".fasta")
    seq = biotite.sequence.io.load_sequence(str(i))
    seq = str(seq)
    uid2seq[uid] = seq
    
uid2seq_len = {uid: len(seq) for uid, seq in uid2seq.items()}
    

# +
direct_interaction_set = {}
bsasa_ref.loc[:, 'hasna'] = (pd.isna(bsasa_ref['Prey1'].values) | pd.isna(bsasa_ref['Prey2'].values))
for label, r in bsasa_ref.loc[~bsasa_ref['hasna'].values, :].iterrows():
    pair = frozenset((r['Prey1'], r['Prey2']))
    direct_interaction_set[pair] = ""
    
nuid = len(set(bsasa_ref['Prey1'].values).union(bsasa_ref['Prey2'].values))

print((f"Of the {h(bsasa_ref.shape[0])} pairs in the BSASA reference\n"
       f"{h(sum(bsasa_ref['hasna'].values))} are NaN\n"
       f"{h(len(set(bsasa_ref['PDBID'].values)))} PDB IDs are represented\n"
       f"{h(len(set(bsasa_ref.loc[~bsasa_ref['hasna'], 'PDBID'])))} PDBIDs after removing NaNs\n"
       f"There are {h(nuid)} Uniprot IDS\n"
       f"There are {h(len(set(bsasa_ref.loc[~bsasa_ref['hasna'].values, 'Prey1'].values).union(bsasa_ref.loc[~bsasa_ref['hasna'].values, 'Prey2'].values)))} Uniprot IDS"
       f" after removing NaNs"))

# +
uid_total = 3062
interaction_total = math.comb(uid_total, 2)
n_possible_mapped_interactions = math.comb(nuid, 2)
n_direct = len(direct_interaction_set)


print(f"Of the {h(uid_total)} total prey {h(interaction_total)} interactions are possible")
print((f"Of these, {h(len(direct_interaction_set))} were found in the PDB\n"
f"representing {h(np.round(100 *len(direct_interaction_set) / interaction_total, 2))}% of interactions"))
print(f"Of the {h(nuid)} mapped prey {h(n_possible_mapped_interactions)} interactions are possible")
print((f"{h(len(direct_interaction_set))} ({np.round((len(direct_interaction_set) / n_possible_mapped_interactions) * 100, 2)}%)"
       f" were found in the PDB"))
print("It is estimated that 0.35%-1.5% of possible protein interactions are positive")
print(f"This corresponds to {h(int(n_possible_mapped_interactions * 0.0035))} to {h(int(n_possible_mapped_interactions * 0.015))} possible interactions")
print(f"The remaining {uid_total - nuid} prey were not found in the PDB")
print(f"This corresponds to {h(math.comb(uid_total - nuid, 2))} possible interactions")
# -

chain_mapping = pd.read_csv("../significant_cifs/chain_mapping_all.csv")

# +
table1 = pd.read_csv("../table1.csv")
print(f"table 1 {table1.shape}")
print(f"BSASA REF {bsasa_ref.shape}")

notna = pd.notna(bsasa_ref["Prey1"].values)
bsasa_ref = bsasa_ref[notna]
notna = pd.notna(bsasa_ref["Prey2"].values)
bsasa_ref = bsasa_ref[notna]
print(bsasa_ref.shape)
pdb_set = set(bsasa_ref['PDBID'].values)
print(f"N PDBS {len(pdb_set)}")
prey_set = set(bsasa_ref['Prey1'].values).union(bsasa_ref['Prey2'].values)
print(f"N Prey {len(prey_set)}")

gene2uid = {key:val for key, val in table1.values}
uid2gene = {val:key for key,val in gene2uid.items()}

print(f"--Prey In BSASA--")
print(f"VIF {gene2uid['vifprotein'] in prey_set}")
print(f"ELOB {gene2uid['ELOB_HUMAN'] in prey_set}")
print(f"ELOC {gene2uid['ELOC_HUMAN'] in prey_set}")
print(f"LRR1 {gene2uid['LLR1_HUMAN'] in prey_set}")
print(f"CBFB {gene2uid['PEBB_HUMAN'] in prey_set}")
print(f"CUL5 {gene2uid['CUL5_HUMAN'] in prey_set}")
print(f"NEDD8 {gene2uid['NEDD8_HUMAN'] in prey_set}")
for i in sorted([f"CSN{i}_HUMAN" for i in range(1, 10)] + ["CSN7A_HUMAN", "CSN7B_HUMAN"]):
    if i in gene2uid:
        print(f"{i.removesuffix('_HUMAN')} {gene2uid[i] in prey_set}")

print(f"--PDBs in BSASA --")

print(f"4n9f {'4n9f' in pdb_set}")

# 4n9f is in the chain mapping vif, CUL5
# BSASA is good, seq_id is good, aln_cols is good
# The interaction between CBFB and Vif is supported by 6p59 C-F
# ELOB interacts with ELOC in 7jto
# ELOB:  Q15370
# ELOC:  Q15369
# 4n9f was not in the dataframe because vif and CUL5 have a BSASA of ~ 380 square angstroms
# ELOB and ELOC were not found in 4n9f because ... hhblits did not find 4n9f for these inputs.

sel = bsasa_ref["Prey1"].values != bsasa_ref["Prey2"].values
bsasa_ref = bsasa_ref[sel]

print(f"\nNon self {bsasa_ref.shape}")
print("-------------------")

plt.hist(bsasa_ref['bsasa_lst'].values, bins=30)
plt.show()

def colprint(name, value):
    print(f"N {name} {value}")

def summarize_col(df, col):
    vals = df[col].values
    colprint(f'{col} min', np.min(vals))
    colprint(f'{col} max', np.max(vals))
    colprint(f'{col} mean', np.mean(vals))
    colprint(f'{col} var', np.var(vals))
    
summarize_col(bsasa_ref, 'bsasa_lst')
    

interaction_set = []
for i, r in bsasa_ref.iterrows():
    interaction_set.append(frozenset((r['Prey1'], r['Prey2'])))
interaction_set = set(interaction_set)

n_possible_interactions = math.comb(3062, 2)
n_possible_found_interactions = math.comb(len(prey_set), 2)
print(f"N possible interactions {h(n_possible_interactions)}")
print(f"N possible found interactions {h(n_possible_found_interactions)}")
print(f"N interactions found {h(len(interaction_set))}")


print('----Per Interaction----')

npdbs_per_interaction = {}

for key in interaction_set:
    prey1, prey2 = key
    sel1 = bsasa_ref['Prey1'] == prey1
    sel2 = bsasa_ref['Prey1'] == prey2
    
    bsasa_sel = bsasa_ref[sel1 | sel2]
    
    sel1 = bsasa_sel['Prey2'] == prey1
    sel2 = bsasa_sel['Prey2'] == prey2
    
    bsasa_sel = bsasa_sel[sel1 | sel2]
    
    pdb_set_sel = set(bsasa_sel['PDBID'])
    
    npdbs_per_interaction[key] = len(pdb_set_sel)
    
print("\nCHAIN MAPPING")
print("---------")

sel = chain_mapping['bt_aln_percent_seq_id'] >= 0.3
print(f"chain_mapping {chain_mapping.shape}")
chain_mapping = chain_mapping[sel]
print(f"30% SeqID {chain_mapping.shape}")

chain_pdb_set = set(chain_mapping['PDBID'].values)
chain_uid_set = set(chain_mapping['QueryID'].values)

print(f"N PDBS {len(chain_pdb_set)}")
print(f"N UIDS {len(chain_uid_set)}")

complexes = {}
for pdb_id in chain_pdb_set:
    sel = chain_mapping['PDBID'] == pdb_id
    chain_mapping_at_pdb = chain_mapping[sel]
    uids = frozenset(chain_mapping_at_pdb['QueryID'].values)
    complexes[pdb_id] = uids
    
cocomplexes = {}
for pdb_id, fset in complexes.items():
    if len(fset) > 1:
        cocomplexes[pdb_id] = fset
        
print(f"N cocomplexes {len(cocomplexes.keys())}")

cocomplex_uid_set = []

for pdb_id, fset in cocomplexes.items():
    for uid in fset:
        cocomplex_uid_set.append(uid)

cocomplex_uid_set = set(cocomplex_uid_set)
cocomplex_pairs = list(combinations(cocomplex_uid_set, 2))
        
cocomplex_df = {pair: "" for pair in cocomplex_pairs}
for pair in cocomplex_pairs:
    for pdb_id, fset in cocomplexes.items():
        p1, p2 = pair
        if (p1 in fset) and (p2 in fset):
            val = cocomplex_df[pair]
            if len(val) == 0:
                val = pdb_id
            else:
                val = val + f";{pdb_id}"
            
            cocomplex_df[pair] = val
            
cocomplex_list = [(key[0], key[1], val) for key, val in cocomplex_df.items()]
cocomplex_df = pd.DataFrame(cocomplex_list, columns=['Prey1', 'Prey2', 'PDBIDS'])

sel = cocomplex_df["PDBIDS"] != ""
cocomplex_df = cocomplex_df[sel]
cocomplex_df.loc[:, 'NPDBS'] = [len(i.split(";")) for i in cocomplex_df['PDBIDS'].values]

print(f"Co complex df {cocomplex_df.shape}")
# -

npairs = math.comb(3062, 2)
nobs = math.comb(1400, 2)
print((f"Of the {h(math.comb(3062, 2))} possible protein interactions"
      f"we expect\n0.35-1.5% ({h(math.comb(3062, 2)*0.0035)}-{h(math.comb(3062,2)*0.015)})\n"
      f"In the BSASA reference set there are {1400} prey\n"
      f"The observable space is less than {h(math.comb(1400, 2))}\n"
      f"Leading to {h(nobs * 0.0035)}-{h(nobs * 0.015)}\n"))


# +
# Two Classes of truth from the Protein Data Bank

# +
vals = cocomplex_df['NPDBS'].values
plt.title("N PDBS per Cocomplex prey pair")
plt.hist(vals, bins=(len(set(vals)) // 1))

plt.show()

# +
# Test the SAINT scores
# Test only bait prey interactions
import seaborn as sns
xlsx_file = "../1-s2.0-S1931312819302537-mmc2.xlsx"
df1 = pd.read_excel(xlsx_file, 0)
df2 = pd.read_excel(xlsx_file, 1)
df3 = pd.read_excel(xlsx_file, 2)

df1 = df1[df1['PreyGene'] != 'IGHG1_MOUSE']
df2 = df2[df2['PreyGene'] != 'IGHG1_MOUSE']
df3 = df3[df3['PreyGene'] != 'IGHG1_MOUSE']

assert "IGHG1_MOUSE" not in df1.values
assert "IGHG1_MOUSE" not in df2.values
assert "IGHG1_MOUSE" not in df3.values

viral_remapping = {
"vifprotein"          :   "P69723",
"polpolyprotein"      :   "Q2A7R5",
"nefprotein"     :        "P18801",
"tatprotein"         :    "P0C1K3",
"gagpolyprotein"     :    "P12493",
"revprotein"          :   "P69718",
"envpolyprotein"      :   "O12164"}

conditions = ["wt_MG132", "vif_MG132", "mock_MG132"]
baits = {"CBFB": "PEBB_HUMAN", "ELOB": "ELOB_HUMAN", "CUL5": "CUL5_HUMAN", "LRR1": "LLR1_HUMAN"}

bait2uid = {}
for bait, genename in baits.items():
    for condition in conditions:
        key = bait + condition
        uid = gene2uid[genename]
        bait2uid[key] = uid

df1.loc[:, "BaitUID"] = [bait2uid[i] for i in df1["Bait"].values]
df2.loc[:, "BaitUID"] = [bait2uid[i] for i in df2["Bait"].values]
df3.loc[:, "BaitUID"] = [bait2uid[i] for i in df3["Bait"].values]

df_all = pd.concat([df1, df2, df3])
df_all.loc[:, "Prey"] = [viral_remapping[i] if i in viral_remapping else i for i in df_all['Prey'].values]

# +
# X axis SAINT score, Y1 Cocomplex interactions, Y2 is 

saint_scores = df_all["SaintScore"].values
direct_interaction_saint = []
direct_interaction = []
direct_interaction_labels = []
cocomplex_interactions = []
cocomplex_interaction_saint = []
cocomplex_interaction_labels = []
unknown = []



cocomplex_pairs = [frozenset((row['Prey1'], row['Prey2'])) for i, row in cocomplex_df.iterrows()]
bsasa_ref_pairs = [frozenset((row['Prey1'], row['Prey2'])) for i, row in bsasa_ref.iterrows()]
for i, row in df_all.iterrows():
    bait_uid = row["BaitUID"]
    prey_uid = row["Prey"]
    
    pair = frozenset((bait_uid, prey_uid))
    found = False
    if pair in cocomplex_pairs:
        cocomplex_interactions.append(1)
        cocomplex_interaction_labels.append(pair)
        found = True
    else:
        cocomplex_interactions.append(0)
    if pair in bsasa_ref_pairs:
        direct_interaction.append(1)
        direct_interaction_labels.append(pair)
        
        found = True
    else:
        direct_interaction.append(0)
    
    if not found:
        unknown.append(1)
    else:
        unknown.append(0)


unknown = np.array(unknown)
direct_interaction = np.array(direct_interaction)
cocomplex_interactions = np.array(cocomplex_interactions)
# -

npairs = math.comb(3062, 2)
nobs = math.comb(1400, 2)
print((f"Of the {h(math.comb(3062, 2))} possible protein interactions"
      f"we expect\n0.35-1.5% ({h(math.comb(3062, 2)*0.0035)}-{h(math.comb(3062,2)*0.015)})\n"
      f"In the BSASA reference set there are {1400} prey\n"
      f"The observable space is less than {h(math.comb(1400, 2))}\n"
      f"Leading to {h(nobs * 0.0035)}-{h(nobs * 0.015)}\n"))


bsasa_ref


# +
def permutation_test(rseed, vals, true, T, n_samples):
    t_true = T(true)
    N = len(true)
    key = jax.random.PRNGKey(rseed)
    sampling = jax.random.choice(key, a=vals, shape=(N, n_samples))
    results = T(sampling, axis=0)
    true_result = T(true)
    return np.array(results), np.array(true_result)
        
    
def plot_sampling(results, true_result, v0, v1, test_stat="", backend='seaborn', title="",
                 vkwargs={'color':'r', 'label': 'True'},
                 histkwargs={'label':'Null'},
                 tx=0,
                 ty=5,
                 ts=f"N samples {''}\nSize {''}", 
                 nbins=None):
    nsamples = len(results)
    
    if not nbins:
        nbins = min(nsamples // 10, 100)
    
    if backend == 'seaborn':
        ax = sns.histplot(results, bins=nbins, **histkwargs)
    elif backend == 'mpl':
        ax = plt.hist(results, bins=nbins, **histkwargs)
    
    ts = ts + f"\np-value {pval(results, true_result)}"
    plt.text(tx, ty, ts)
    plt.vlines(true_result, v0, v1, **vkwargs)
    plt.xlabel(test_stat)
    plt.legend()
    plt.title(title)
    plt.show()
    
    return ax
    
def pval(results, true_results):
    N = len(results)
    return np.sum(results >= true_result) / N


# +
import jax
vals = df_all['SaintScore'].values

results, true_result = permutation_test(13, vals, vals[np.where(direct_interaction)], np.mean, 1000000)

ax = plot_sampling(results, true_result, 0, 250000, 
                   test_stat="T(x): Mean Saint Score of 9 bait-prey pairs", 
                   title="Permutation test",
                   tx=0.05,
                   ty=240000,
                   ts=f"N samples w/ replacement {h(len(results))}\nSize {sum(direct_interaction)}",
                   nbins=30,
                   histkwargs={'label': 'Null'})

print("H0: Direct Bait-Prey interactions have no relation to SaintScore")

# +
vals = df_all['SaintScore'].values

results, true_result = permutation_test(13, vals, vals[np.where(cocomplex_interactions)], np.mean, 1000000)
ax = plot_sampling(results, true_result, 0, 130000, 
                   test_stat="T(x): Mean Saint Score of 19 bait-prey pairs", 
                   title="Permutation test",
                   tx=0.05,
                   ty=120000,
                   ts=f"N samples w/ replacement {h(len(results))}\nSize {sum(cocomplex_interactions)}",
                   nbins=30,
                   histkwargs={'label': 'Random pair'})

print("H0: Cocomplex bait-prey interactions have no relation to Saint Score")

# +
# N unknown
benchmark_summary = pd.DataFrame([len(cocomplex_df), len(bsasa_ref), len(df_all),
                                 sum(unknown), sum(direct_interaction), sum(cocomplex_interactions),
                                 math.comb(3062, 2)], 
                                 columns=["N"], index=['PDB Co-complex', 'PDB Direct', 'Bait-prey',
                                                      'Bait-prey absent in PDB',
                                                      'Bait-prey Direct',
                                                      'Bait-prey cocomplex',
                                                      'Possible Interactions'])


cols = ['Co-complex', 'Direct', 'Bait-prey']
sns.categorical.barplot(benchmark_summary.T.iloc[:, 0:4])
plt.ylabel('N Interactions')
# -

benchmark_summary

sns.categorical.barplot(benchmark_summary.T.iloc[:, 4:6])
plt.ylabel('N Interactions')

# +
# Load in the xarrays and scores
from functools import partial


df_all.index = [i.split("_")[0] for i in df_all["PreyGene"]]
conditions = ['wt', 'vif', 'mock']
bait = ['CBFB', 'ELOB', 'CUL5', 'LRR1']
Bait2bait = {}
Bait2condition = {}
for key in bait:
    for val in conditions:
        Bait2bait[key + val + '_MG132'] = key
        Bait2condition[key + val + '_MG132'] = val
        

        
df_all.loc[:, 'bait'] = [Bait2bait[i] for i in df_all['Bait'].values]
df_all.loc[:, 'condition'] = [Bait2condition[i] for i in df_all['Bait'].values]

rsel = [f"r{i}" for i in range(1, 5)]
csel = [f"c{i}" for i in range(1, 13)]
df_all = parse_spec(df_all)
df_new = df_all[['bait', 'condition', 'Prey', 'SaintScore', 'BFDR'] + rsel + csel]

prey_set = sorted(list(set(df_new.index)))
preyu = np.array(prey_set)
preyv = np.array(prey_set)
nprey = len(preyu)

edge_list = list(combinations(preyu, 2))

cocomplex_matrix = xr.DataArray(np.zeros((nprey, nprey)), 
                                coords={"preyu": preyu, "preyv": preyv}, dims=["preyu", "preyv"])
direct_matrix = xr.DataArray(np.zeros((nprey, nprey)), 
                            coords={"preyu": preyu, "preyv": preyv}, dims=["preyu", "preyv"])

# Bait, condition, preyu, r, c

# Create a tensor filled with zeros

# expeRiment tensor
tensorR = [[[[0 for _ in rsel] for _ in range(len(preyu))] for _ in range(len(bait))]
           for _ in range(len(conditions))]

# Control tensor
tensorC = [[[[0 for _ in csel] for _ in range(len(preyu))] for _ in range(len(bait))]
          for _ in range(len(conditions))]

bait2idx = {b:i for i, b in enumerate(bait)}
condition2idx = {c:i for i, c in enumerate(conditions)}
r2idx = {r:i for i, r in enumerate(rsel)}
c2idx = {c:i for i, c in enumerate(csel)}
prey2idx = {p:i for i,p in enumerate(preyu)}

def fill_tensors(df, tensorR, tensorC, condition_name, bait_mapping, prey_mapping, condition_mapping,
                r_cols, c_cols, prey_col_name, bait_col_name="Bait"):
    # Fill the tensor with the 'C' value where 'A' and 'B' intersect
    for _, row in df.iterrows():
        bait_name = row[bait_col_name].split("_")[0][0:4] # The All bait are 4 characters long
        bait_index = bait_mapping[bait_name]
        prey_index = prey_mapping[row[prey_col_name]]
        condition_index = condition_mapping[condition_name]
        tensorR[condition_index][bait_index][prey_index] = [row[c] for c in r_cols]
        tensorC[condition_index][bait_index][prey_index] = [row[c] for c in c_cols]
        
    return tensorR, tensorC

df_new.loc[:, 'PreyName'] = df_new.index

fill_tensors_applied = partial(fill_tensors, bait_mapping=bait2idx,
                              prey_mapping=prey2idx, condition_mapping=condition2idx,
                              r_cols=rsel, c_cols=csel, prey_col_name='PreyName', bait_col_name="bait")

s1 = df_new['condition'] == 'wt'
s2 = df_new['condition'] == 'vif'
s3 = df_new['condition'] == 'mock'

tensorR, tensorC = fill_tensors_applied(df=df_new[s1], tensorR=tensorR, tensorC=tensorC, condition_name="wt")

tensorR, tensorC = fill_tensors_applied(df=df_new[s2], tensorR=tensorR, tensorC=tensorC, condition_name="vif")

tensorR, tensorC = fill_tensors_applied(df=df_new[s3], tensorR=tensorR, tensorC=tensorC, condition_name="mock")



for b in bait:
    for c in conditions:
        sel1 = df_new['bait'] == b
        sel2 = df_new['condition'] == c
        dtemp = df_new[sel1 & sel2]
        print(dtemp.shape, b, c)


tensorC = xr.DataArray(np.array(tensorC, dtype=int),
                      dims=['condition', 'bait', 'preyu', 'crep'],
                      coords={'condition': conditions,
                             'bait': bait,
                             'preyu': preyu,
                             'crep': np.arange(0, 12)})

tensorR = xr.DataArray(np.array(tensorR, dtype=int),
                      dims=['condition', 'bait', 'preyu', 'rrep'],
                      coords={'condition': conditions,
                             'bait': bait,
                             'preyu': preyu,
                             'rrep': np.arange(0, 4)})


preyname2uid = {row['PreyName']:row['Prey'] for i,row in df_new.iterrows()}
uid2preyname = {val:key for key,val in preyname2uid.items()}
cocomplex_df.loc[:, 'Prey1Name'] = [uid2preyname[i] for i in cocomplex_df['Prey1'].values]
cocomplex_df.loc[:, 'Prey2Name'] = [uid2preyname[i] for i in cocomplex_df['Prey2'].values]

for i, r in cocomplex_df.iterrows():
    p1 = r['Prey1Name']
    p2 = r['Prey2Name']
    val = r['NPDBS']
    
    cocomplex_matrix.loc[p1, p2] = val
    cocomplex_matrix.loc[p2, p1] = val

    
bsasa_ref.loc[:, 'Prey1Name'] = [uid2preyname[i] for i in bsasa_ref['Prey1'].values]
bsasa_ref.loc[:, 'Prey2Name'] = [uid2preyname[i] for i in bsasa_ref['Prey2'].values]

for i, r in bsasa_ref.iterrows():
    p1 = r['Prey1Name']
    p2 = r['Prey2Name']
    val = direct_matrix.loc[p1, p2].values
    
    direct_matrix.loc[p1, p2] = val + 1
    direct_matrix.loc[p2, p1] = val + 1


# Populate the co


ds = xr.Dataset({'cocomplex': cocomplex_matrix, 'direct': direct_matrix, 'CRL_E':tensorR, 'CRL_C':tensorC})
# -

ds

df_new[df_new['condition']=='wt']

kws = {'stat':'probability'}
sns.histplot(saint_scores, label=f"All N: {len(saint_scores)}", bins=10, **kws)
sns.histplot(saint_scores[np.where(direct_interaction)], 
             label=f"Direct {sum(direct_interaction)}", bins=10, alpha=0.5, **kws)
sns.histplot(saint_scores[np.where(cocomplex_interactions)], 
             label=f"Cocomplex {sum(cocomplex_interactions)}", bins=10, alpha=0.5, **kws)
plt.legend()
plt.xlabel("Saint Score")
plt.show()

sns.histplot(saint_scores, label=f"All N: {len(saint_scores)}", bins=10, **kws)
plt.show()

saint_scores[np.where(direct_interaction)]

for (i,j) in direct_interaction_labels:
    print(uid2gene[i], uid2gene[j])

for (i,j) in cocomplex_interaction_labels:
    print(uid2gene[i], uid2gene[j])

df_all['Spec']

sns.histplot(unknown)

np.sum(unknown == 0)

vals = np.array([len(v) for k,v in cocomplexes.items()])
plt.hist(vals, bins=(len(set(vals)) // 1))
plt.title("N Protein types per cocomplex")
plt.show()

# +
vals = np.array([val for key,val in npdbs_per_interaction.items()])
plt.title("N PDBS per Interaction")
plt.hist(vals, bins=20)
plt.show()


# -

chain_mapping['bt_aln_percent_seq_id'].values

bsasa_ref

chain_mapping

# +
# Direct Interactions benchmark

# +
prey_pairs = list(combinations(prey_set, 2))
direct_interaction_set = {}
for i, r in bsasa_ref.iterrows():
    p1 = r['Prey1Name']
    p2 = r['Prey2Name']
    direct_interaction_set[frozenset((p1, p2))] = ""
pdb_positive = []
prey1 = []
prey2 = []
for pair in prey_pairs:
    fs = frozenset(pair)
    p1, p2 = pair
    prey1.append(p1)
    prey2.append(p2)
    if fs in direct_interaction_set:
        pdb_positive.append(True)
    else:
        pdb_positive.append(False)
        
prey_pairs_df = pd.DataFrame({'Prey1Name': prey1, 'Prey2Name': prey2, 'pdb_pos': pdb_positive})
nrows, ncols = prey_pairs_df.shape

prey_pairs_df.loc[:, 'rand'] = np.array(jax.random.uniform(jax.random.PRNGKey(13), shape=(nrows, )))
# -

# DataSet
print(f"N direct {h(np.sum(np.tril(ds['direct'] > 0, k=-1)))}")
print(f"N cocomplex {h(np.sum(np.tril(ds['cocomplex'] > 0, k=-1)))}")

# +
# Append Cocomplex labels to DataFrame

cocomplex_labels = []
for i, r in df_new.iterrows():
    bait = r['bait']
    prey = r['PreyName']
    if bait == 'LRR1':
        bait = 'LLR1'
    elif bait == 'CBFB':
        bait = 'PEBB'
        
    val = ds['cocomplex'].sel(preyu=bait, preyv=prey).item()
    cocomplex_labels.append(val)
    
df_new.loc[:, 'PDB_COCOMPLEX'] = np.array(cocomplex_labels, dtype=bool)
    
    

# +
df_new


# -

ds.sel(preyu='PEBB')

xr.DataArray(df_new.sort_index())

# +
import operator
# Function for plotting an accuracy curve
def npos_ntotal(prey_pairs_df, col, threshold, comp=operator.le, pos_col='pdb_pos'):
    sel = prey_pairs_df[col] <= threshold
    sub_df = prey_pairs_df.loc[sel, :]
    npos = np.sum(sub_df[pos_col].values)
    ntotal = len(sub_df)
    return npos, ntotal

def xy_from(prey_pairs_df, col, thresholds, comp, pos_col):
    npos = []
    ntot = []
    for t in thresholds:
        p, nt = npos_ntotal(prey_pairs_df, col, t, comp=comp, pos_col=pos_col)
        npos.append(p)
        ntot.append(nt)
    return np.array(ntot), np.array(npos)


# +
x, y = xy_from(df_new, 'SaintScore', np.arange(1, 0, -0.05), comp=operator.ge, pos_col='PDB_COCOMPLEX')

pos_col = 'PDB_COCOMPLEX'
npairs = len(df_new)
npdb_pos = sum(df_new[pos_col].values)

yplot = y / npdb_pos
xplot = x / npairs
plt.plot(xplot, yplot, 'k.', label='Saint Score')
plt.ylabel(f"Fraction PDB Cocomplex Positives (N={h(npdb_pos)})")
plt.xlabel(f"Fraction Total Positives (N={h(npairs)})")

#y2 = 1.0
#x2 = npdb_pos / npairs
#plt.vlines([0.0035, 0.015], 0, 1, label="Estimated fraction of true PPIs")


#plt.plot(x2, y2, 'r+', label='PDB Benchmark')
##xmul = 10
#plt.plot(x2 * xmul, y2, 'rx', label=f'{xmul}x PDB')
#plt.savefig('f1.png', dpi=300)
plt.legend()
plt.show()
# -

df_new.loc[:, 'rAv'] = df_new.loc[:, rsel].mean(axis=1)
df_new.loc[:, 'cAv'] = df_new.loc[:, csel].mean(axis=1)
df_new.loc[:, 'Av_diff'] = df_new['rAv'].values - df_new['cAv'].values

df_new.shape

# +
"""
JSON Structure for PreFilter

There is 1 control shared between wt and vif conditions
There is 1 control for the mock

- vif/wt ctrl
- mock ctrl
- wt
- vif
- mock
-rsel (4)
-csel (12)

vif - vif/wt ctrl
wt  - vig/wt ctrl
mock - mock ctrl
"""

def df_new2json(df):
    csel = [f"c{i}" for i in range(1, 13)]
    rsel = [f"r{i}" for i in range(1, 5)]
    
    Nrows = len(df)
    Nrsel = len(rsel)
    Ncsel = len(csel)
    
    RMatrix = df.loc[:, rsel].values
    CMatrix = df.loc[:, csel].values
    
    RMatrix = [[int(i) for i in row] for row in RMatrix]
    CMatrix = [[int(i) for i in row] for row in CMatrix]
    
    return {'Nrsel': Nrsel, 'Ncsel': Ncsel, 'Nrows': Nrows, 'Yexp': RMatrix,
           'Yctrl': CMatrix}


# -

prey2seq = {r['QueryID']: r['Q'] for i, r in chain_mapping.iterrows()}
df_new.loc[:, 'Q'] = np.array([(prey2seq[prey] if prey in prey2seq else np.nan) for prey in df_new['Prey'].values])

xt = np.arange(-10, 10)
yt = np

plt.title("Experimental Counts")
x = np.ravel(df_new.loc[:, rsel].values)
plt.hist(x, bins=100)
s = f"Mean {np.mean(x)}\nMedian {np.median(x)}\nVar {np.var(x)}\nMin {np.min(x)}\nMax {np.max(x)}"
plt.text(100, 30000, s)
plt.show()

df_new.loc[:, 'rVar'] = df_new.loc[:, rsel].var(axis=1).values
df_new.loc[:, 'cVar'] = df_new.loc[:, csel].var(axis=1).values

# +
import scipy as sp

x1 = df_new['rAv'].values
y1 = df_new['rVar'].values
x2 = df_new['cAv'].values
y2 = df_new['cVar'].values

r1, p1 = sp.stats.pearsonr(x1, y1)
r2, p2 = sp.stats.pearsonr(x2, y2)

r1 = np.round(r1, 2)
r2 = np.round(r2, 2)

plt.plot(x1, y1, 'k.', label='experiment')
plt.plot(x2, y2, 'rx', label='control')
s = f"R Experiemnt {r1}\nR control {r2}"
plt.text(60, 1400, s)
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.legend()
plt.show()


# -

ctrl_counts = (df_new.loc[(df_new['condition']=='wt') | (df_new['condition']=='mock'), csel])
x = np.ravel(ctrl_counts.values)
plt.title("Control Counts")
plt.hist(x, bins=100)
s = f"Mean {np.mean(x)}\nMedian {np.median(x)}\nVar {np.var(x)}\nMin {np.min(x)}\nMax {np.max(x)}"
plt.text(100, 30000, s)
plt.show()

# +
xall = np.concatenate([np.ravel(ctrl_counts.values), np.ravel(df_new.loc[:, rsel].values)])
bounds = (0, 50)
xaxis = np.arange(0, bounds[1], 1)
r = 1/5
yvalue = r * np.exp(-r * xaxis) * 5e5

plt.hist(xall, bins=50, range=bounds)
plt.plot(xaxis, yvalue, 'r', label='Model')
plt.title("All Counts")
s = f"Mean {np.mean(xall)}\nMedian {np.median(xall)}\nVar {np.var(xall)}\nMin {np.min(xall)}\nMax {np.max(xall)}"
plt.xlabel("Spectral Count")
plt.legend()
plt.text(100, 30000, s)
plt.show()

def exp_pdf(x, r):
    return r * np.exp(-x * r)


# -

seq_lens = np.array([uid2seq_len[prey] for prey in df_new['Prey'].values])
df_new.loc[:, 'aa_seq_len'] = seq_lens
df_new.loc[:, 'exp_aa_seq_len'] = np.exp(seq_lens)


# +
def n_tryptic_cleavages(aa_seq):
    assert aa_seq.isupper()
    aa_seq = np.array(list(aa_seq))
    Ksites = aa_seq == 'K'
    Rsites = aa_seq == 'R'
    Allsites = Ksites | Rsites
    n_sites = np.sum(Allsites)
    if (aa_seq[-1] == 'K') or (aa_seq[-1] == 'R'):
        n_sites -= 1
    
    return n_sites

def n_first_tryptic_cleavages(aa_seq):
    return 2 * n_tryptic_cleavages(aa_seq)

def n_first_typtic_cleavage_peptides(aa_seq):
    return 1 + n_first_tryptic_cleavages(aa_seq)


# -

n_first_sites = [n_first_tryptic_cleavages(uid2seq[prey]) for prey in df_new['Prey']]
df_new.loc[:, 'n_first_tryptic_cleavage_sites'] = np.array(n_first_sites)
df_new.loc[:, 'n_possible_first_tryptic_peptides'] = df_new.loc[:, 'n_first_tryptic_cleavage_sites'] + 1

sel = df_new['aa_seq_len'] <= 5000
sns.histplot(df_new[sel], x='n_first_tryptic_cleavage_sites', y='aa_seq_len')

sns.regplot(df_new[sel], x='rAv', y='n_first_tryptic_cleavage_sites')

sel = df_new['aa_seq_len'] <= 200
sns.regplot(df_new[sel], x='cAv', y='n_first_tryptic_cleavage_sites')

sns.regplot(df_new[sel], x='rAv', y='aa_seq_len')

plt.plot(np.arange(len(df_new)), df_new['rAv'] - df_new['cAv'], 'k.')

plt.plot(np.arange(len(df_new)), df_new['rAv'], 'k.')

plt.plot(np.arange(len(df_new)), df_new['cAv'], 'k.')

# +
nbins=30
aa_max_len=2000
xlabel='n_possible_first_tryptic_peptides'
#xlabel='aa_seq_len'
sns.histplot(df_new[df_new['aa_seq_len'] <=aa_max_len], x=xlabel, y='rAv',
            bins=(nbins, nbins), cbar=True, cmap='hot')

print("The sequence length appears to set the upper bound on Spectral Count")
print("The protein spectral count is the sum of peptide spectral count")
print(f"Proteins are well sampled in the {np.min(df_new[xlabel].values), np.mean(df_new[xlabel].values)} {xlabel} range")
print(f"At least two possibilities")
print(f"1. Longer sequences are cleaved into more peptides")
print("2. Longer sequences fragment into more fragment ions")
print(f"3. Both")
print(f"The result of this effect is that longer sequences are more detectable")
print(f"Therefore longer sequences may be have lower abundance at lower abundance ")
# -

sns.histplot(df_new[df_new['aa_seq_len'] <=aa_max_len], x=xlabel, y='cAv',
            bins=(nbins, nbins), cbar=True, cmap='hot')

sns.histplot(df_new[df_new['aa_seq_len'] <=aa_max_len], x=xlabel, y='rMin',
            bins=(nbins, nbins), cbar=True, cmap='hot')



# There is one sequence that is very long
col = 'aa_seq_len'
sel = df_new[col] < 5000
sns.regplot(df_new.loc[sel, :], x=col, y='rAv')
print("It appears like the sequence length places an upper bound on the spectral counts")

col = 'aa_seq_len'
xcol = 'rAv'
sel = df_new[col] <= 500
plt.plot((df_new.loc[sel, col].values), np.max(df_new.loc[sel, rsel].values, axis=1), 'k.')
plt.xlabel(col)
plt.ylabel('Max spectral counts')
#plt.ylabel(col)

col = 'exp_aa_seq_len'
sel = df_new[col] < 1.4e217
sns.regplot(df_new[sel], x=col, y='rAv')

# +
"""
95 % Frequentist confidence interval
"""

x = np.arange(0, 215)
b = 0
m = 1.565
y = m * x + b
sns.regplot(x="rAv", y="rVar", data=df_new)
plt.plot(x, y)
plt.title("Experimental Data")

# +
x = np.arange(0, 215)
b = 0
m = 2.8
y = m * x + b

ax = sns.regplot(x='cAv', y="cVar", data=df_new)
plt.plot(x, y)
plt.title("Control Data")


# -

def model(x=None, y=None):
    b = 0 #numpyro.sample('b', dist.Normal(0, 0.1))
    m = numpyro.sample('m', dist.Normal(0, 1))
    numpyro.sample('Y', dist.Normal(m * x + b), obs=y)



nuts_kernal = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=500, num_samples=1000)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, x=df_new['rAv'].values, y=df_new['rVar'].values, extra_fields=('potential_energy',))

mcmc.print_summary()


# +
def m1(df):
    nrows = len(df)
    hyper_prior = np.ones(nrows) * 1/5
    Yexp = df.loc[:, rsel].values
    Yctrl = df.loc[:, csel].values
    
    nexp = len(rsel)
    nctrl = len(csel)
    
    kappa_ = numpyro.sample('k', dist.Exponential(rate=hyper_prior))
    lambda_ = numpyro.sample('l', dist.Exponential(rate=hyper_prior))
    for i in range(0, nctrl):
        numpyro.sample(f'Ycrtl_{i}', dist.Poisson(kappa_), obs=Yctrl[:, i])
        
    for i in range(0, nexp):
        numpyro.sample(f'Yexp_a{i}', dist.Poisson(lambda_), obs=Yexp[:, i])
        numpyro.sample(f'Yexp_b{i}', dist.Poisson(kappa_), obs=Yexp[:, i])
    
    
    
# -

df_test = df_new
nuts_kernal = numpyro.infer.NUTS(m1)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=1000)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, df_test, extra_fields=('potential_energy',))

# ## Probabalistic Filter
#
# The model is
# $$ p(M | D, I) \propto p(D | M, I)p(M | I) $$
#
# $$ D = Y_E, Y_C $$
# $$ I = I_1, I_2 $$
#
#

samples = mcmc.get_samples()

# +
# First create a function that reads in a vector of

jax.vmap(jax.scipy.stats.poisson.pmf)


# +
def f1(Y, x):
    return 

def posterior_odds(samples, df):
    theta1 = samples['l']
    theta2 = samples['k']
    
    Yexp = df[rsel].values
    f = jax.scipy.stats.poisson.pmf
    
    p1 = f(Yexp, mu=theta1.T)
    p2 = f(Yexp, mu=theta2.T)
    return p1, p2
    
    
# -

x1 = samples['k'].T[:, 0]
y1 = df_new.loc[:, rsel].values


# +
def f(x, Yexp):
    """
    x as a vector, return average probability
    """
    return jax.vmap(jax.scipy.stats.poisson.pmf)(Yexp, x).sum(axis=1) / 4

def f2(X, Yexp):
    nrows, nsamples = X.shape
    return jax.vmap(f, in_axes=[1, None])(X, Yexp).sum(axis=0) / nsamples

def posterior_odds(samples, Yexp):
    K = samples['k'].T
    L = samples['l'].T
    
    return np.array(f2(K, Yexp)), np.array(f2(L, Yexp))

Yexp = df_new.loc[:, rsel].values
odds_kappa, odds_lambda = posterior_odds(samples, Yexp)

# -

odds_kappa[np.where(odds_kappa==0)] = 1e-7

plt.hist(odds_kappa, bins=100)
plt.show()

plt.hist(odds_lambda, bins=100)
plt.show()

log_odds_ratio = np.log10(odds_lambda) - np.log10(odds_kappa)
# Capped at -4 and 4
log_odds_ratio[np.where(log_odds_ratio >= 4)] = 4
log_odds_ratio[np.where(log_odds_ratio <= -4)] = -4
print(np.min(log_odds_ratio), np.max(log_odds_ratio))

capped_odds_ratio = 10**log_odds_ratio

plt.hist(capped_odds_ratio, bins=100, range=(0,10))
plt.xlabel("Odds Ratio")
plt.show()

plt.hist(log_odds_ratio, bins=100, range=(-5, 5))
plt.xlabel("Log Odds Ratio M1 vs M2")
plt.show()


# +
# Impact of selecting a threshold
def thresh_sel(t, x):
    """
    Return the number of remaining entries
    
    """
    return len(x[x >= t])

thresholds = np.arange(-5, 5, 0.1)
remaining = []
for i in thresholds:
    remaining.append(thresh_sel(i, log_odds_ratio))

remaining = np.array(remaining)
plt.plot(thresholds, remaining)
plt.xlabel('Log10 odds threshold')
plt.ylabel('Data Remaining')


# +
def bait_box_plot(ds, var='CRL_E', preysel=True, boxkwargs={}):
    vals = []
    z = [('CBFB', 'PEBB'), ('ELOB', 'ELOB'), ('CUL5', 'CUL5')]#, ('LRR1', 'LLR1')]
    for i in z:
        if preysel:
            arr = np.ravel(ds.sel(bait=i[0], preyu=i[1])[var].values)
        else:
            arr = np.ravel(ds.sel(bait=i[0])[var].values)
        vals.append(arr)
    
    vals.append(arr)
    z.append(('LRR1', 'LLR1'))
    arr = np.ravel(ds.sel(bait='LRR1', preyu='LLR1', condition='mock')[var].values)
    if var == 'CRL_E' and preysel==False:
        arr = np.ravel(ds.sel(bait='ELOB')['CRL_C'].values) # Bait can be any because control is similiar
        vals.append(arr)
        z.append('Parent')
        

    
    sns.boxplot(vals, **boxkwargs)#, labels=['A'] * 4)
    plt.title("Amount of Bait in own purification across 3 conditions")
    plt.xticks(np.arange(len(vals)), [i[0] for i in z])
    plt.ylabel("Spectral count")
    plt.show()
    
bait_box_plot(ds)

# -

bait_box_plot(ds, var='CRL_C')

bait_box_plot(ds, preysel=False, boxkwargs={'ylim': 50})


# +
def slice_up_df_metric(df, thresholds, col, compare_f, action_f):
    results = []
    for t in thresholds:
        sel = compare_f(df[col].values, t)
        results.append(action_f(df[sel]))
        
    return results

def corr_action(df, col1, col2):
    return sp.stats.pearsonr(df[col1].values, df[col2])

corr_mean_var_action = partial(corr_action, col1='rAv', col2='rVar')

def n_remaining_action(df):
    return len(df)


thresholds = np.arange(0, 211.75, 0.1)

corr = slice_up_df_metric(df_new, thresholds, 'rAv', operator.le, corr_mean_var_action)
rcorr, pval = zip(*corr)
n_remaining = slice_up_df_metric(df_new, thresholds, 'rAv', operator.le, n_remaining_action)


# +
def compare_window_f(a, N, b):
    return (N - b < a) & (a < N + b)
        
        
def simple_scatter(x, y, xname, yname, title=None):
    plt.plot(x, y, 'k.')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    


# -

wf = partial(compare_window_f, b=30)
corr_list = slice_up_df_metric(df_new, thresholds, 'rAv', wf, corr_mean_var_action)
n_remaining_win = slice_up_df_metric(df_new, thresholds, 'rAv', wf, n_remaining_action)
rcorr_win, pval_win = zip(*corr_list)

plt.plot(n_remaining, rcorr, 'k.')
plt.xlabel("N datapoints remaining")
plt.ylabel("Mean variance correlation")

simple_scatter(x=rcorr, y=pval, xname='Pearson R', yname='P-val')



simple_scatter(x=thresholds, y=rcorr, xname='SC threshold', yname='Pearson R')

# +
# Low abundance
sel = df_new['rAv'] <= 200
sel2 = df_new['bait'] != 'LRR1'
sel = sel & sel2
x = df_new.loc[sel, 'rAv'].values
y = df_new.loc[sel, 'rVar'].values

simple_scatter(x, y, xname='rAv', yname='rVar')
# -

df_new.loc[:, 'rMax'] = np.max(df_new.loc[:, rsel].values, axis=1)
df_new.loc[:, 'rMin'] = np.min(df_new.loc[:, rsel].values, axis=1)
df_new.loc[:, 'cMax'] = np.max(df_new.loc[:, csel].values, axis=1)
df_new.loc[:, 'cMin'] = np.min(df_new.loc[:, csel].values, axis=1)

# +
aa_max_len = 1000
sns.histplot(df_new[df_new['aa_seq_len'] <= aa_max_len], 
             x='n_possible_first_tryptic_peptides', 
             y='cMax',
             cmap='hot')

print("This plot can be used to formulate a data likelihood")
print("The likelihood of the Max spectral count | n_first_tryptic_peptides")
print("We are interested in protein abundance")
print("We need p(Max SC | abundance, n_first_tryptic_peptides)")
print("Specifically we are interested in the abundance at a specific time")
print("Not the abundance at the detector")
print("We are interested in the abundance of the peptide in the mixture")
print("If we assume an unkown abundance distribution (can place a prior)")
print("That is indpendent of N possible first tryptic peptides")
print("")

# +
sns.regplot(df_new, x='rAv', y='rMax')
m=1.15
y = df_new['rAv'].values * m
x=df_new['rAv'].values

plt.plot(x, y, 'r')
# -

sns.regplot(df_new, x='rAv', y='rMin')

sns.regplot(df_new, x='cAv', y='cMax')

sns.regplot(df_new, x='cAv', y='cMin')

sns.regplot(df_new, x='rMin', y='rMax')


plt.plot(df_new['rAv'], df_new['rMax'].values - df_new['rMin'], 'k.')

sns.regplot(df_new, x='cMin', y='cMax')



sel = df_new['rAv'] <= 12
sns.regplot(df_new[sel], x='rAv', y='rMax')

sel1 = df_new['cAv'] <=10
sel2 = df_new['bait'] != 'LRR1'
sel = sel1 & sel2
sns.regplot(df_new[sel], x='cAv', y='cVar')

df_new.loc[sel, csel + ['cAv']].sort_values('cAv', ascending=False).iloc[150:200, :]

plt.plot(np.arange(0, 30), jax.scipy.stats.poisson.pmf(np.arange(0, 30), 15))

# +
sel2 = df_new['cAv'] <= 5
sns.regplot(df_new[sel2], x='cAv', y='cMax')
x=df_new.loc[sel2, 'cAv']
m=4
y = m * x

plt.plot(x, y, 'r')
# -

y2 = np.max(df_new.loc[sel, rsel].values, axis=1)
simple_scatter(x, y2, xname='rAv', yname='rMax')
m=1
y = m * x
plt.plot(x, y, 'r')

y2

df_new

simple_scatter(x=thresholds, y=n_remaining, xname='SC threshold', yname='N remaining')

plt.hist(pval_win, bins=50)
plt.show()

simple_scatter(x=thresholds, y=rcorr_win, xname='SC threshold', yname='Win R')

simple_scatter(x=thresholds, y=n_remaining_win, xname='SC threshold', yname='N remaining')



sp.stats.pearsonr

operator.lt(df_new['rAv'].values, 2)

bait_box_plot(ds, preysel=False, var='CRL_C')


def abundance_over_all_conditions(ds):
    vals = []
    for 


# +
# Check the double counting of the condition
x = ds['CRL_C']
k = 'CUL5'
k2 = 'ELOB'
x2 = x.sel(bait=k, condition='wt')

x2[np.any((x.sel(bait=k, condition='vif').values != x.sel(bait=k2, condition='wt').values), axis=1)]
# -

x.sel(bait='CUL5')


# +
def model2(ds, N=3062):
    
    # wt, vif, mock
    # 
    # [condition, bait, prey, rrep]
    
    ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_E'].values
    CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_E'].values
    CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_E'].values
    
    ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_E'].values
    CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_E'].values
    CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_E'].values
    
    ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_E'].values
    CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_E'].values
    CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_E'].values
    
    LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_E'].values
    
    ctrl_ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_C'].values
    
    ctrl_ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_C'].values
    
    ctrl_ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_C'].values
    
    ctrl_LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_C'].values
    
    
   # max_val = ds['CRL_E'].max('rrep')
    
   # mu_Nc = np.ones((5, 3))
   # mu_alpha = np.ones((N, 5, 3))
    

    
    #N = numpyro.sample('N', dist.Normal(np.zeros(3), 5))
    #mu = numpyro.sample('mu', dist.Normal(max_val.sel(bait='ELOB').values.T, 50), sample_shape=(3062, 3))
    #numpyro.sample('sc', dist.Normal(N * mu), obs=max_val.sel(bait='ELOB').values.T)
    
    
    
    
    #N1 = numpyro.sample('N1', dist.Normal(0, 1))
    #N2 = numpyro.sample('N2', dist.Normal(0, 1))
    
    #mu_elob = numpyro.sample('mu_elob', dist.Normal(np.mean(ELOB_wt, axis=1), np.var(ELOB_wt, axis=1)))
    #mu_cul5 = numpyro.sample('mu_cul5', dist.Normal(np.mean(CUL5_wt, axis=1), np.var(ELOB_wt, axis=1)))
    
    #numpyro.sample('ELOB_wt', dist.Normal(mu_elob * N1, 5), obs=ELOB_wt)
    #numpyro.sample('CUL5_wt', dist.Normal(mu_cul5 * N2, 5), obs=CUL5_wt)
    
    
    #cell_abundance = numpyro.sample(dist.Normal(jnp.ones((3, 5))), 1)
    
    assert ELOB_wt.shape == (3062, 4)
    
    mu_hyper_prior = np.ones((3062, 1)) / 50
    sig_hyper_prior = np.ones((3062, 1)) / 2
    
    
    mu = numpyro.sample('mu', dist.Exponential(mu_hyper_prior))
    sigma = numpyro.sample('s', dist.Exponential(sig_hyper_prior))
    
    Ncells = numpyro.sample('Nc', dist.Normal(np.ones((1, 4)), 0.5))
    
    Ncells_rep = jnp.repeat(Ncells, 3062, axis=0)
    
    
    numpyro.sample('sc', dist.Normal(mu * Ncells_rep, sigma), obs=ELOB_wt)
    
    #Ncells = cell_abundance * 1e7 
    
    #gamma_i = numpyro.sample('gamma', dist.Beta(0.5, 0.5), sample_shape=(3062,))
    #mu_ctrl = numpyro.sample('mu0', dist.Uniform(0, 250), sample_shape=(3062,))
    #mu_wt = numpyro.sample('mu_wt', dist.Uniform(0, 250), sample_shape=(3062,))
    
    #numpyro.sample('ELOB_wt', dist.Normal(mu_wt, 10), obs=ELOB_wt)
    #numpyro.sample('ctrl_ELOB_wt', dist.Normal(mu_ctrl * gamma_i, 10), obs=ctrl_ELOB_wt)
    
    
print(f"N free parameters {5 * 3 + 5 * 3 * 3062}")  
# -

df

x = np.arange(0, 1, 0.01)
y = jax.scipy.stats.beta.pdf(x, a=2 ,b=6)
y = jax.scipy.stats.expon.pdf(x* 250, 100)
plt.plot(x * 300, y, 'k.')

# +
plt.hist(np.ravel(np.std(df_new[rsel].values, axis=1)), bins=50, density=True)

x = np.arange(0, 30, 0.1)
y = np.exp(dist.Exponential(1/2).log_prob(x)) * 1.3
plt.plot(x, y, 'r')
# -

# ?dist.Normal

x = np.arange(0, 300)
plt.plot(x, np.exp(dist.Exponential(1/50).log_prob(x)))

sum(df_new['Av_diff'] < -10)

sns.scatterplot(df_new, x='Av_diff', y='SaintScore')

plt.hist(np.ravel(ds.sel(bait=["CBFB", "ELOB", "CUL5"])['CRL_E'].values), bins=100)
plt.show()

plt.hist(np.ravel(ds.sel(bait=["CBFB", "ELOB", "CUL5"])['CRL_C'].values), bins=100)
plt.show()


def model4(e_data=None, ctrl_data=None, a=None, b=None):

    
    # Prior around 1/5
    if a is None:
        a = numpyro.sample('a', dist.Uniform(0.0001, 1))
    if b is None:
        b = numpyro.sample('b', dist.Uniform(0.0001, 1))
    
    
    numpyro.sample('y_e', dist.Exponential(a), obs=e_data)
    numpyro.sample('y_c', dist.Exponential(b), obs=ctrl_data)

# +
dsel = ds.sel(bait=['CBFB', 'ELOB', 'CUL5'])
ctrl_data = np.ravel(dsel['CRL_C'].values)
e_data = np.ravel(dsel['CRL_E'].values)

nuts_kernal = numpyro.infer.NUTS(model4)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=10000, thinning=2)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, e_data=e_data, ctrl_data=ctrl_data, extra_fields=('potential_energy',))
# -

mcmc.print_summary()

posterior_samples = mcmc.get_samples()

posterior_predictive = numpyro.infer.Predictive(model4, posterior_samples)(jax.random.PRNGKey(1))



posterior_predictive['y_c'].shape

prior = numpyro.infer.Predictive(model4, num_samples=1000)(jax.random.PRNGKey(2))

m4_data = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive)

az.plot_trace(m4_data['sample_stats'], var_names=['lp'])

prior.keys()

prior['y_c'].shape

plt.hist(prior['y_c'], bins=100, label='prior control', alpha=0.5)
plt.hist(prior['y_e'], bins=100, label='prior experiment', alpha=0.5)
plt.hist(posterior_predictive['y_c'], bins=100, label='posterior control')
plt.hist(posterior_predictive['y_e'], bins=100, label='posterior experiment', alpha=0.5)
plt.show()



plt.hist(posterior_samples['a'], bins=50, label='Experiment')
plt.hist(posterior_samples['b'], bins=50, label='Control')
#plt.hist(prior['a'], bins=50)
plt.title(f"Posterior N={len(posterior_samples['a'])}")
plt.xlabel("Exponential rate")
plt.legend()
plt.show()

# +
# Rate 

x = np.arange(0, 300)
y = np.exp(dist.Exponential(1/80).log_prob(x))
plt.scatter(x, y)
# -

x = np.arange(0, 400)
y = np.exp(dist.HalfNormal(200).log_prob(x))
sns.scatterplot(x=x, y=y)
plt.show()


def model5(a=None, b=None, y_c=None, y_e=None, poisson_sparse=True, batch_shape=()):
    """
    a:  experimental rate
    b:  control rate
    
    y_e: experimental obs
    y_c: control obs
    """
    
    #max_val = np.max([np.max(y_c), np.max(y_e)])
    
    # Assum the true rates are within 10 + 2 * max observed value
    
    
    
    hyper = np.ones(batch_shape) * 200 #* max_val
    
    if a is None:
        a = numpyro.sample('a', dist.HalfNormal(hyper))
    if b is None:
        b = numpyro.sample('b', dist.HalfNormal(hyper))
    
    # predictive checks
    nrows, ncols = batch_shape
    
    if y_c is None:
        b = jnp.ones((nrows, 12)) * b
    
    if y_e is None:
        a = jnp.ones((nrows, 4)) * a
        
    
    numpyro.sample('y_c', dist.Poisson(b, is_sparse=poisson_sparse), obs=y_c)
    numpyro.sample('y_e', dist.Poisson(a, is_sparse=poisson_sparse), obs=y_e)

nuts_kernal = numpyro.infer.NUTS(model4)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=10000, thinning=2)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, e_data=e_data, ctrl_data=ctrl_data, extra_fields=('potential_energy',))


def do_mcmc(model, rng_key, 
            model_kwargs, 
            Kernel=numpyro.infer.NUTS,
            Search=numpyro.infer.MCMC, 
            num_warmup=500,
            num_samples=1000,
            thinning=1,
            extra_fields=('potential_energy',)):
    

    search_kwargs = {'num_warmup': num_warmup, 
                    'num_samples': num_samples,
                    'thinning': thinning}
    
    run_kwargs = {'extra_fields': extra_fields}
    
    search_kwargs = {} if search_kwargs is None else search_kwargs
    run_kwargs = {} if run_kwargs is None else run_kwargs
    
    kernel = Kernel(model)
    search = Search(kernel, **search_kwargs)
    
    
    search.run(rng_key, **model_kwargs, **run_kwargs)
    
    
    return search

model = model5
n = 0
y_c = np.array(df_new.iloc[n, :][csel].values, dtype=int)
y_e = np.array(df_new.iloc[n, :][rsel].values, dtype=int)
model_kwargs={'y_c': y_c, 'y_e': y_e}
search = do_mcmc(model5, 1, model_kwargs, num_warmup=1000, num_samples=5000)

search.print_summary()

posterior_samples = search.get_samples()

prior = numpyro.infer.Predictive(model5, num_samples=500)

posterior_predictive = numpyro.infer.Predictive(model5, posterior_samples)


def multi_hist(hists, bins, labels, alphas, xlabel):
    for i in range(len(hists)):
        plt.hist(hists[i], bins=bins, label=labels[i], alpha=alphas[i])
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()


from numpyro.diagnostics import hpdi, summary

# +
hists = [posterior_samples['a'], posterior_samples['b'], posterior_samples['a'] - posterior_samples['b']]
labels = ['Experiment', 'Control', 'Difference']
alphas = [0.8, 0.8, 0.3]

multi_hist(hists, bins=50, labels=labels, alphas=alphas, xlabel='Poisson rate')
# -

posterior_samples['a'].shape

from collections import namedtuple


def outer_f(x):
    def inner_r(x):
        return _secret_f(x)
    
    def _secret_f(x):
        print(f"Hello, {x}!")
        
    return inner_r


inner_r = outer_f(None)

# +

start = 0
end = 100

def center_and_scale_predictor(y, shift=2):
    assert y.ndim == 2
    col_shape = (len(y), 1)
    mean, var = np.mean(y, axis=1).reshape(col_shape), np.var(y, axis=1).reshape(col_shape)
    var[var==0] = 1
    
    y = ((y - mean) / var) + shift
    return namedtuple('ParamScale', 'y mean var shift col_shape')(
        y, mean, var, shift, col_shape
    )

def un_center_and_scale_predictor(x, param_scale, min_value=1e-8, safe=True):
    if safe:
        assert x.shape[0] == param_scale.col_shape[0], (x.shape, param_scale.col_shape)
    x = np.array(x)
    var = param_scale.var
    var[var==0] = 1
    
    x = (x - param_scale.shift) * var + param_scale.mean
    x[np.where(x <= min_value)] = 0.
    return x
    
    

    


def model52_f(df_new, 
              start, 
              end, 
              numpyro_model = model5,
              numpyro_model_kwargs = None):
    
    if numpyro_model_kwargs is None:
        numpyro_model_kwargs = {}
    
    def init_kernel(rescale_model=False):
        n = slice(start, end)
        dsel = df_new.iloc[n, :]
        l = len(dsel)

        y_c = np.array(dsel[csel].values, dtype=int)
        y_e = np.array(dsel[rsel].values, dtype=int)

        #col_shape = (l, 1)

        # Scale and shift predictors
        #mean_c, var_c = np.mean(y_c, axis=1).reshape(col_shape), np.var(y_c, axis=1).reshape(col_shape)
        #mean_e, var_e = np.mean(y_e, axis=1).reshape(col_shape), np.var(y_e, axis=1).reshape(col_shape)


        #var_c[var_c==0]=1
        #var_e[var_e==0]=1

        #y_c = ((y_c - mean_c) / var_c) + 2.
        #y_e = ((y_e - mean_e) / var_e) + 2.
        
        if rescale_model:
            y_c_param_scale = center_and_scale_predictor(y_c)
            y_e_param_scale = center_and_scale_predictor(y_e)
            model_kwargs={'y_c': y_c_param_scale.y, 'y_e': y_e_param_scale.y}
            
        else:
            model_kwargs={'y_c': y_c, 'y_e': y_e}
            y_c_param_scale = None
            y_e_param_scale = None
            
        model_meta={'y_c': y_c_param_scale, 'y_e': y_e_param_scale}
        
        model = partial(
            numpyro_model, 
            batch_shape=(l, 1),
            **numpyro_model_kwargs) # partial apply for prior and post pred checks
        
        return namedtuple('Init', 'model kwargs meta')(model, model_kwargs, model_meta)
    
    def init_sampling(rng_key, num_warmup, num_samples):
        return namedtuple('Init', 'rng_key num_warmup num_samples')(rng_key, num_warmup, num_samples)
    
    def sample(model_init, sample_init):
        search = do_mcmc(model_init.model, 
                         sample_init.rng_key, 
                         model_init.kwargs,
                         num_warmup = sample_init.num_warmup,
                         num_samples = sample_init.num_samples)
        #samples = search.get_samples()
        #summary_dict = summary(samples, group_by_chain=False)
        return search
    
    
    #k1, k2, k3 = jax.random.split(rng_key, 3)
    #search = do_mcmc(model, k1, model_kwargs, num_warmup=1000, num_samples=5000)
    #samples = search.get_samples()
    #summary_dict = summary(samples, group_by_chain=False)
    
    def sample_pp(model_init, sample_init):
        """
        Sample from the prior predictive distribution
        """
        return numpyro.infer.Predictive(
            model = model_init.model, 
            num_samples = sample_init.num_samples
        )(sample_init.rng_key)
    
    def sample_Pp(model_init, samples, sample_init):
        return numpyro.infer.Predictive(
            model = model_init.model, 
            posterior_samples = samples
        )(rng_key = sample_init.rng_key)
    
    def _pre_init_InferenceData(
        search,
        pp,
        Pp,
        model_meta
    ):
    
        coords = {'irow': np.arange(start, end), 'col': np.array([0]),
                  'rrep': np.arange(4), 'crep': np.arange(12)}

        dims = {'a': ['chain', 'draw', 'irow', 'col'],
                'b': ['chain', 'draw', 'irow', 'col'],
                'y_e': ['irow', 'rrep'], 
                'y_c':['irow', 'crep']}
        

        pred_dims = {'y_c': ['chain', 'draw', 'irow', 'col'],
                     'y_e': ['chain', 'draw', 'irow', 'col']}
            
        
        inf_data = az.from_numpyro(search, 
            prior=pp, posterior_predictive=Pp, coords=coords, dims=dims, pred_dims=pred_dims)
        assert inf_data is not None
        return inf_data
        
    def rescale_model(model_meta, inf_data):

        # Scale Observed
        yobs = inf_data['observed_data']
        
        inf_data.observed_data['y_c'].values = un_center_and_scale_predictor(
            yobs.y_c, model_meta['y_c'])
        
        inf_data.observed_data['y_e'].values = un_center_and_scale_predictor(
            yobs.y_e, model_meta['y_e'])
        
        # Scale Posterior
        
        ufunc = parital(un_center_and_scale_predictor, param_scale=model_meta['y_e'])
        
        inf_data.posterior.a = xr.apply_ufunc(
            ufunc, inf_data.posterior.a)
        
        
        
        inf_data.posterior.a.values = un_center_and_scale_predictor(
            inf_data.posterior.a)


        return inf_data
    
    
    def _append_posterior_statistics(
        inf_data, 
        samples, 
        group_by_chain=False
    ):
        summary_dict = summary(samples, group_by_chain=group_by_chain)
        
        a_rhat = summary_dict['a']['r_hat'][:, 0]
        a_neff = summary_dict['a']['n_eff'][:, 0]
        b_rhat = summary_dict['b']['r_hat'][:, 0]
        b_neff = summary_dict['b']['n_eff'][:, 0]
        
        tmp_df = pd.DataFrame({'a_rhat': a_rhat, 'a_neff': a_neff, 'b_rhat':b_rhat, 'b_neff': b_neff})
        inf_data['posterior']['stats'] = xr.DataArray(tmp_df.values, 
                                                    coords={'irow': np.arange(start, end),
                                                    'stat': np.array(tmp_df.columns)})
        return inf_data
    
    def init_InferenceData(
        search,
        pp,
        Pp,
        model_meta,
        rescale=True,
        append_sample_stats=True,
        group_by_chain=False
    ):
        inf_data = _pre_init_InferenceData(search, pp, Pp, model_meta)
        
        if rescale:
            inf_data = rescale_model(model_meta = model_meta, inf_data = inf_data)
        if append_sample_stats:
            inf_data = _append_posterior_statistics(inf_data, search.get_samples(), group_by_chain=group_by_chain)
            
        return inf_data
    
    return namedtuple(
        "M5F", "init_kernel init_sampling sample sample_pp sample_Pp rescale_model init_InferenceData")(
            init_kernel = init_kernel,
            init_sampling = init_sampling,
            sample = sample,
            sample_pp = sample_pp,
            sample_Pp = sample_Pp,
            rescale_model = rescale_model,
            init_InferenceData = init_InferenceData
        )

def run_and_merge_models(df_new, from_, to, step, rng_key):
    posterior = []
    prior = []
    observed = []
    prior_p = []
    post_p = []
    log_like = []
    sample_stats = []
    
    
    intervals = list(range(from_, to, step))
    for i in range(len(intervals)):
        start = intervals[i]
        end = start + step
        print(start ,end)
        rng_key, k1 = jax.random.split(rng_key)
        
        m_data = model52m_data(df_new, start, end, k1)
        
        posterior.append(m_data.posterior)
        prior.append(m_data.prior)
        observed.append(m_data.observed_data)
        prior_p.append(m_data.prior_predictive)
        log_like.append(m_data.log_likelihood)
        sample_stats.append(m_data.sample_stats)
    
    posterior = xr.merge(posterior)
    prior = xr.merge(prior)
    observed = xr.merge(observed)
    prior_p = xr.merge(prior_p)
    post_p = xr.merge(post_p)
    log_like = xr.merge(log_like)
    sample_stats = xr.merge(sample_stats)
    
    
    return av.InferenceData({'posterior': posterior, 'prior': prior, 'posterior_predictive': post_p,
                            'prior_predictive': prior_p, 'log_likelihood': log_like,
                            'observed_data': observed})
    
    


# -

def model6(hyper_a, hyper_b, a=None, b=None, y_c=None, y_e=None, poisson_sparse=True, batch_shape=()):
    """
    Similiar to model 5 however we place stronger priors on Poisson
    rates in the hopes of speeding up HMC
    """
    
    if a is None:
        a = numpyro.sample('a', dist.HalfNormal(hyper_a))
    if b is None:
        b = numpyro.sample('b', dist.HalfNormal(hyper_b))
    
    # predictive checks
    nrows, ncols = batch_shape
    
    if y_c is None:
        b = jnp.ones((nrows, 12)) * b
    
    if y_e is None:
        a = jnp.ones((nrows, 4)) * a
        
    
    numpyro.sample('y_c', dist.Poisson(b, is_sparse=poisson_sparse), obs=y_c)
    numpyro.sample('y_e', dist.Poisson(a, is_sparse=poisson_sparse), obs=y_e)

import arviz as az

# +
# Variants - number of chains
# Keep the model as a variable
start=0
end=1000#len(df_new)
l = end - start
y_c = df_new.iloc[start:end, :][csel].values
y_e = df_new.iloc[start:end, :][rsel].values

kd = {'hyper_a': (np.mean(y_e, axis=1).reshape((l, 1)) + 10) * 1.5,
      'hyper_b': (np.mean(y_c, axis=1).reshape((l, 1)) + 10) * 1.5}

# +
m5f = model52_f(df_new, start=start, end=end, numpyro_model=model6,
               numpyro_model_kwargs=kd)

kernel = m5f.init_kernel(rescale_model=False)


sample_init = m5f.init_sampling(jax.random.PRNGKey(13), num_warmup=1000, num_samples=5000)
search = m5f.sample(kernel, sample_init)

# +
sample_init = m5f.init_sampling(PRNGKey(12), num_warmup=1000, num_samples=1000)
pp = m5f.sample_pp(kernel, sample_init)
sample_init = m5f.init_sampling(PRNGKey(11), num_warmup = 1000, num_samples = 1000)
Pp = m5f.sample_Pp(kernel, search.get_samples(), sample_init)

inf_data = m5f.init_InferenceData(search, pp, Pp, kernel.meta, rescale=False, append_sample_stats=True)
# -

inf_data

# +
#ufunc = partial(un_center_and_scale_predictor, param_scale = kernel.meta['y_e'], safe=False)
#inf_data.posterior['a'] = xr.apply_ufunc(ufunc, inf_data.posterior.a)

# +
#az.plot_trace(inf_data, var_names=['a'], plot_kwargs={'xlim': (0, 10)})

# +
#yobs = inf_data.observed_data

# +
#models = run_and_merge_models(df_new, 0, 20, 10, jax.random.PRNGKey(13))
# -

"""
Let's say you have roughly 20,000 independant parameters.
How many should fall outside the prior and posterior predictive checks

"""


# +
# Analysis
def summary_stats(m_data):
    data = m_data.posterior.stats.sel(stat=['a_rhat', 'b_rhat', 'a_neff', 'b_neff'])
    max_ = data.max('irow').values
    min_ = data.min('irow').values
    
    std = data.std(dim='irow').values
    med = data.median('irow').values
    
    return pd.DataFrame([max_, min_, med, std], 
                        index=['max', 'min', 'med', 'std'], 
                        columns=data.stat.values)

def plot_lp(m_data):
    az.plot_trace(m_data.sample_stats['lp'])

def plot_prior(m_data, start=0, end=10):
    az.plot_trace(m_data.prior.sel(irow=np.arange(start, end)))
    
def predictive_check(m_data, T='max', 
                     map_axis_pair=('y_e', 'rrep')):
    
    var, dim = map_axis_pair
    Tobs = (m_data.observed_data[var]).max(dim).values
    Tprior = (m_data.prior_predictive[var]).max(dim).values
    Tpost = (m_data.posterior_predictive[var]).max(dim).values
    
    prior_n, *_ = plt.hist(np.ravel(Tprior), bins=10, label='Prior predictive', alpha=0.8)
    post_n, *_ = plt.hist(np.ravel(Tpost), bins=10, label='Posterior predictive', alpha=0.8)
    ymax = max(np.mean(prior_n), np.mean(post_n))
    plt.vlines(Tobs.item(), 0, ymax, 'k', label='observed')
    plt.xlabel(f"T(y) : {T} spectral count")
    
    plt.legend()

summary_stats(inf_data)
plot_lp(inf_data)

# +
axes = az.plot_trace(inf_data.sample_stats['lp'])
ax = axes[:, 0].item()
ax.hist(np.ravel(inf_data.sample_stats['lp'].values), bins=100, color='C1', alpha=0.5)
ax.grid()
ax.set_ylabel("Frequency")

plt.show()

# +
fig, axs = plt.subplots(nrows=1, ncols=2)
alphas = [0.2, 0.2]
for i, key in enumerate(['a', 'b']):
    axs[i].plot(inf_data.posterior.stats.sel(stat=f"{key}_rhat").values,
                inf_data.posterior.stats.sel(stat=f"{key}_neff").values, 'k.', alpha=alphas[i])
    axs[i].set_xlabel(f"{key} Rhat")
    axs[i].set_ylabel(f"{key} Neff")
    axs[i].set_ylim((3000, 8000))
    axs[i].set_xlim((0.999, 1.001))
fig.tight_layout()


#inf_data.posterior.stat

# +
def _get_summary_stats(y_sim, dims=('rrep', 'crep')):
    return namedtuple('T', "mean min max var")(
            y_sim.mean(dim=dims),
            y_sim.min(dim=dims),
            y_sim.max(dim=dims),
            y_sim.var(dim=dims))

def ds_get_summary_stats(ds, dims=('rrep', 'crep')):
    return _get_summary_stats(ds, dims=dims)

def inf_get_summary_stats(inf_data, chain_num=0):
    return (ds_get_summary_stats(inf_data.prior_predictive.sel(chain=chain_num)),
            ds_get_summary_stats(inf_data.posterior_predictive.sel(chain=chain_num)),
            ds_get_summary_stats(inf_data.observed_data))

def check_satisfaction(sym_s, obs_s, prob=0.9, 
    var_names=['y_c', 'y_e'],
    dim='draw'):

    """
    Does the distribution satisfy the observed data?
    """

    a = sym_s.min <= obs_s.min
    b = sym_s.max >= obs_s.max

    hdi_mean = ds_hdi(sym_s.mean, prob = prob, var_names = var_names, dim = dim)
    hdi_var  = ds_hdi(sym_s.var,  prob = prob, var_names = var_names, dim = dim)
    
    s1 = hdi_mean.sel(hdi='min') <= obs_s.mean
    s2 = hdi_mean.sel(hdi='max') >= obs_s.mean
    
    mean_sat = s1 & s2
    
    s1 = hdi_var.sel(hdi='min') <= obs_s.var
    s2 = hdi_var.sel(hdi='max') >= obs_s.var
    
    var_sat = s1 & s2



    return namedtuple('satisfaction', 'min max mean var')(
        a, b, mean_sat, var_sat
    )

def satisfaction(inf_data, chain_num=0, dims=('rrep', 'crep')):
    """
    Define Data Satisfaction in a simple way

    - a dataset satifies the model if
      - the obs mean is within the sample mean
      - the obs variance is within the sample variance
      - the sim min value <= obs min value
      - the sim max value >= obs max value
    """
    
    y_pp_sim = inf_data.prior_predictive.sel(chain=chain_num)
    y_Pp_sim = inf_data.posterior_predictive.sel(chain=chain_num)
    
    pp_s, Pp_s, o_s = inf_get_summary_stats(inf_data, chain_num=chain_num)
    
    
    return namedtuple("pc", "pp Pp obs")(
        check_satisfaction(pp_stats, obs_stats), 
        check_satisfaction(Pp_stats, obs_stats),
        obs_stats
    )


def ds_hdi(x: xr.Dataset, var_names, dim, prob=0.9):
    
    axes = [x[name].get_axis_num(dim) for name in var_names]
    assert len(np.unique(axes)) == 1, axes
    axis_num = axes[0]
    f = partial(da_hdi, dim=dim, prob=prob)
    return x.map(f)
    

def da_hdi(x: xr.DataArray, dim, prob):
    axis_num = x.get_axis_num(dim)
    y = hpdi(x.values, axis=axis_num, prob=prob)
    hdi = np.array(['min', 'max'])
    coords = {'hdi': hdi, 'irow': x.coords['irow']}
    return xr.DataArray(data=y, coords=coords, dims=['hdi', 'irow'])


# Shared dimensions: irow
# Shared variable names, y_c 

# Passing Frame

def to_satisfaction_frame(p_c):
    """
    sat tup: A tuple from check_satsifaction
    """
    
    df = pd.DataFrame(index=p_c.mean.coords['irow'], data={
        'mean_c': p_c.mean.y_c,
        'mean_e': p_c.mean.y_e,
        'var_e':  p_c.var.y_e,
        'var_c':  p_c.var.y_c
    })
    df.loc[:, 'all_e'] = np.alltrue(df.loc[:, ['mean_e', 'var_e']], axis=1)
    df.loc[:, 'all_c'] = np.alltrue(df.loc[:, ['mean_c', 'var_c']], axis=1)
    return df.loc[:, ['mean_c', 'var_c', 'all_c', 'mean_e', 'var_e', 'all_e']]


# +
# Data Satisfaction
# -

sat = satisfaction(inf_data)

prior_sat = to_satisfaction_frame(sat.pp)
post_sat = to_satisfaction_frame(sat.Pp)

alpha=0.9
(prior_sat.sum() / len(prior_sat)).plot(kind='bar', label='prior', alpha=alpha)
(post_sat.sum() / len(post_sat)).plot(kind='bar', label='posterior', color='C1', alpha=alpha)
plt.title("Data satisfaction of predicted checks")
plt.grid()
plt.legend()

# ?sns.barplot

prior_sat

pc = satisfaction(inf_data)

sat.pp.mean

ppc = pd.DataFrame(index=inf_data.posterior.coords['irow'].values, columns=['min', 'max', 'var', 'mean'])

ppc

sat = satisfaction(inf_data)

sat.pp.min.y_c

sat.pp

tmp[0][0]

ds_hpdi(inf_data.posterior_predictive.sel(chain=0).mean(dim=['crep', 'rrep']), 
        var_names=['y_c', 'y_e'], dim_name='draw')



# ?np.unique

np.unique([1, 2, 1, 1])

# ?xr.DataArray

inf_data.posterior_predictive.sel(chain=0).mean(dim=['crep', 'rrep']).map(f)

f(inf_data.posterior_predictive.sel(chain=0).mean(dim=['crep', 'rrep']).y_c, dim_name='draw')

# ?inf_data.posterior.map

hpdi(np.arange(0, 100).reshape((20, 5)), axis=1)

inf_data.posterior_predictive.sel(chain=0).mean(dim=['rrep', 'crep']).reduce(hpdi, dim=['draw'])

inf_data.posterior_predictive.sel(chain=0).mean(dim=['rrep', 'crep']).coords

# ?hpdi

pps, Pps = satisfaction(inf_data)

pps[0].mean(dim=['draw'])

pp_stats, Pp_stats, obs_stats = satisfaction(inf_data)

pp_stats.mean - obs_stats.mean

inf_data.observed_data.mean(dim=['crep', 'rrep'])

plot_prior(m_data, start=0, end=10)

# +
# Compute the HDPI 
# -

predictive_check(m_data.sel(irow=0))

# +
q = np.array([0.1, 0.5, 0.9])
q_func = partial(np.quantile, q=q, axis=0)
q_func(m_data.prior_predictive.sel(chain=0)['y_c'].reduce(np.max, dim='crep')).T


# Prior predictive
# min, max, var, median
# prior, post, obs

test_stats = {'min': np.min, 'max': np.max, 'var': np.var, 'med': np.median}

data_sets = {'pp': m_data.prior_predictive, 'Pp': m_data.posterior_predictive, 'o': m_data.observed_data}

summary_stats = {}


for Tname, T in test_stats.items():
    for dname, data_set in data_sets.items():
        if dname != 'o':
            s = data_set.sel(chain=0).reduce(T, dim=['crep', 'rrep'])
        else:
            s = data_set.reduce(T, dim=['crep', 'rrep'])
        for key in ['y_e', 'y_c']:
            skey = dname + "_" + Tname + "_" + key
            if dname != 'o':
                summary_stats[skey] = q_func(s[key])
            else:
                summary_stats[skey] = s[key].values
        


# -

def out_of_distribution_score(upper, lower, observed):
    length = upper - lower
    a = np.min([lower, observed], axis=0) - lower
    b = np.max([upper, observed], axis=0) - upper
    
    return np.abs(a + b)# / length


def ood_from_summary_stats(summary_stats, T = 'min', pred='post', y_i='y_c'):
    if pred == 'post':
        p = 'Pp_'
    elif pred == 'prior':
        p = 'pp_'
    else:
        raise ValueError
    
    key = p + T + "_" + y_i
    okey = 'o_' + T + "_" + y_i
    x = summary_stats[key].T
    lower, median, upper = x[:, 0], x[:, 1], x[:, 2]
    observed = summary_stats[okey]
    y = out_of_distribution_score(upper, lower, observed)
    return y


def plot_from_summary_stats(summary_stats, T, pred, y_i):
    y = ood_from_summary_stats(summary_stats, T, pred, y_i)
    x = np.arange(len(y))
    plt.plot(x, y, 'k.')
    plt.ylabel('Out distr')


plot_from_summary_stats(summary_stats, 'var', 'prior', 'y_c')

plt.errorbar(np.arange(len(median)), median)

"""
20,000 samples
"""

# +
min_o = m_data.observed_data.reduce(np.min, dim=['crep', 'rrep'])
min_pp = m_data.prior_predictive.reduce(np.min, dim=['crep', 'rrep']).sel(chain=0).min(dim='draw')


# -

x = np.arange(1000)
plt.plot(x, (min_o - min_pp).sortby('y_c')['y_c'].values, '.', label='Prior Predictive Control')
plt.plot(x,)
#plt.plot(x, (min_o - min_pp).sortby('y_c')['y_e'].values, '.', label='Preior Predictive AP')
plt.ylim(-4, 4)
plt.ylabel("Min Obs - min D")
plt.grid()

# +
min_o = m_data.observed_data.reduce(np.min, dim=['crep', 'rrep'])
#max_o = m_data.observed_data.reduce(np.max, dim=['crep', 'rrep'])

min_pp = m_data.prior_predictive.reduce(np.min, dim=['crep', 'rrep']).sel(chain=0).min(dim='draw')
min_Pp = m_data.posterior_predictive.reduce(np.min, dim=['crep', 'rrep']).sel(chain=0).min(dim='draw')

a = (min_o - min_pp).sortby('y_c')['y_c'].values
b = (min_o - min_pp).sortby('y_c')['y_e'].values
c = (min_o - min_Pp)['y_c'].values
d = (min_o - min_Pp)['y_e'].values



def ppc_boxen(boxes, 
              labels=['ppc control', 'ppc AP', 'PpC control', 'PpC AP'], font_dict = {'size': 16},
              title=f"T(y) - T({y_tilde})"):

    y_tilde = '\u1EF9'
    sns.boxenplot(boxes)
    plt.xticks(np.arange(len(boxes)),labels , **font_dict)
    plt.title(title, **font_dict)
    plt.ylabel("Spectral Count Difference", **font_dict)
    #plt.text(2.5, 50, "T: Min", **font_dict)
    plt.grid()
ppc_boxen([a, b, c, d])
#plt.ylim(-1, 1)

# +
"""
HPDI: Narrowest Interval with probability mass of prob


Is it in the narrowest interval with 0.9 probability mass?
Is it in the narrowest interval 

1. Apply Test statistic
2. Compute HPDI at row
3. Check 

"""

def in_hpdi(x, arr, prob, hpdi_fun=hpdi):
    i = hpdi_fun(arr, prob)
    if i[0] < x < i[1]:
        return True
    else:
        return False


# +
def in_hpdi(prob, m_data, dataset, ysel, repsel, chainsel, t):
    T_of_pp = m_data[dataset][ysel].sel(chain=chainsel).reduce(t, dim=repsel)
    hpdi_of_T = hpdi(T_of_pp, prob=prob, axis=0).T
    T_obs = m_data.observed_data[ysel].reduce(t, dim=repsel)
    a = hpdi_of_T[:, 0] <= T_obs
    b = T_obs < hpdi_of_T[:, 1]
    in_interval = a & b
    return in_interval


def hpdi_check(interval_probs, m_data, dataset, ysel, repsel, chainsel, t):
    """
    Return the Highest Posterior Density Interval with the smallest probability mass
    """
    shape = (m_data[dataset].dims['irow'], len(interval_probs))
    results = np.ones(shape) * 2
    for i in range(shape[1]):
        in_intervals = in_hpdi(prob=interval_probs[i], m_data=m_data, dataset=dataset,
                              ysel=ysel, repsel=repsel, chainsel=chainsel,
                              t=t)
        
        results[in_intervals, i] = interval_probs[i]
    results = np.min(results, axis=1)
    results[results==2] = -0.1
    return results



# +
thresholds = np.arange(0.999, 0.01, -0.005)
results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.max)

def hpdi_cdf(results, title, hist_kwargs={}):
    results[results==[-0.1]]=1.2
    plt.title(title)
    plt.hist(results, cumulative=True, bins=100, density=True, **hist_kwargs)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(which='both')
    plt.xlim(0, np.max(results[results <= 1]))
    plt.xlabel("Minimal HPDI")
    plt.ylabel("Probability Mass")
    
hpdi_cdf(results, title="Prior Predictive T(x): Max")
# -

results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.min)

hpdi_cdf(results, title='Prior Predictive T(x): Min')

# +
results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.var)

r2 = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_e', 'rrep', 0, np.var)
# -

hpdi_cdf(results, title='Prior Predictive T(x): Var', hist_kwargs={'label': 'control'})
plt.hist(r2, label='AP', cumulative=True, density=True, alpha=0.5, bins=100)
plt.xlim(0, 0.8)
plt.legend()
plt.show()

results = hpdi_check(thresholds, m_data, 'posterior_predictive', 'y_c', 'crep', 0, np.var)

r2 =      hpdi_check(thresholds, m_data, 'posterior_predictive', 'y_e', 'rrep', 0, np.var)

hpdi_cdf(results, title='Prior Predictive T(x): Var', hist_kwargs={'label': 'control'})
plt.hist(r2, label='AP', cumulative=True, density=True, alpha=0.5, bins=100)
plt.xlim(0, 0.8)
plt.legend()
plt.show()

"""
Conclusions


"""

results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.mean)

results[results==[-0.1]]=1.2
plt.title(f"T: Mean")
plt.hist(results, cumulative=True, bins=100, density=True)
plt.grid()
plt.xlim(0, np.max(results[results <= 1]))
plt.xlabel("Minimal HPDI")
plt.ylabel("Probability Mass")

plt.plot(np.arange(1000), sorted(results, reverse=False), 'k.')

# ?np.sort

a = in_hpdi(0.9999, m_data, dataset='prior_predictive', ysel='y_c', repsel='crep', chainsel=0, t=np.max)

m_data['posterior'].dims['irow']



in_interval

m_data.observed_data.reduce(lambda x, axis: hpdi(x, axis=axis), dim=['crep', 'rrep'])

hpdi(m_data.observed_data['y_c'].values, axis=1)



# What decile does the data fall in?
hpdi()

# +
max_o = m_data.observed_data.reduce(np.max, dim=['crep', 'rrep'])
#max_o = m_data.observed_data.reduce(np.max, dim=['crep', 'rrep'])

max_pp = m_data.prior_predictive.reduce(np.max, dim=['crep', 'rrep']).sel(chain=0).max(dim='draw')
max_Pp = m_data.posterior_predictive.reduce(np.max, dim=['crep', 'rrep']).sel(chain=0).max(dim='draw')

a = (max_pp - max_o).sortby('y_c')['y_c'].values
b = (max_pp - max_o).sortby('y_c')['y_e'].values
c = (max_Pp - max_o)['y_c'].values
d = (max_Pp - max_o)['y_e'].values
# -

ppc_boxen([a, b], ['Control', 'AP'], title="Max: T(y~) - T(y)")

ppc_boxen([c, d], ['Control', 'AP'])

# +
# Magnitude of the deviation outside the 
# -

ticks = np.hstack([np.arange(i) for i in [4, 4]]).reshape((2, 4))

"$y \sim $"

# ?min_o.sortby

m_data.reduce(np.min, dim=['crep', 'rrep'])

plt.plot(np.arange(20000), np.arange(20000), 'k.')

# +

plt.plot(np.arange(len(y)), sorted(y, reverse=True), 'k.')
plt.ylabel('Out of distribution score')
# -

np.min([lower, observed], axis=0).

# ?np.min



# +
#1 Does the observed value fall within the 1st and 9th decile of the prior?
#2 Does the observed value fall within the 1st and 9th decile of the posterior pred?
#3 How much did the model learn?

# What are the most uncertain distributions?


summary_stats['pp_min_y_e'].T - summary_stats['o_min_y_e'].reshape((1000, 1))
# -

summary_stats['pp_min_y_e'].T

summary_stats['o_min_y_e'].reshape((1000, 1))

q_func(s[key])

m_data.prior_predictive.sel(chain=0).reduce(np.min, dim=['crep', 'rrep'])

q_func(m_data.prior_predictive.sel(chain=0)['y_c'].reduce(np.min, dim='crep')).T

m_data.observed_data.reduce(np.min, dim=['crep', 'rrep'])



# ?np.mean

# +

#m_data.posterior_predictive.sel(chain=0)['y_c'].reduce(np.mean, dim='crep').reduce(q_func, dim='draw')
# -

np.quantile(np.arange(10), [0.3, 0.6])

# ?np.quantile

m_data.posterior_predictive

hpdi(m_data.posterior['a'].values, axis=1)

predictive_check(m_data.sel(irow=0))

# +




# -

m_data.posterior_predictive['y_c'].mean('crep')

xr.apply_ufunc(np.mean, m_data.posterior)

# ?xr.apply_ufunc

m_data.prior_predictive

m_data.prior_predictive['y_c'].shape

neff_rhat = {'a_r_hat': summary_dict['a']['r_hat']}



m_data



m_data

m_data

m_data

samples = search.get_samples()

summary_dict = summary(samples, group_by_chain=False)

summary_dict['a'].keys()


def mcmc_rank_plot(summary_dict, stat='r_hat'):
    rhats = []
    for key in summary_dict:
        r = list(np.ravel(summary_dict[key][stat]))
        rhats = rhats + r
    
    rhats = sorted(rhats, reverse=True)
    plt.scatter(np.arange(len(rhats)), rhats)
    plt.ylabel(stat)
    plt.xlabel('Param Rank')
    


mcmc_rank_plot(summary_dict, stat='n_eff')

mcmc_rank_plot(summary_dict)

hpdi(posterior_samples['a'] - posterior_samples['b'])

plt.hist(posterior_samples['a'], bins=100, label='Experimental rate')
plt.hist(posterior_samples['b'], bins=100, label='Control rate')
plt.hist(posterior_samples['a'] - posterior_samples['b'], bins=100, label='Rate difference')
plt.legend()
plt.show()

# +

y= np.exp(dist.HalfCauchy(200).log_prob(x))
sns.scatterplot(x=x, y=y)
plt.show()
# -

posterior_predictive['y_c'][-5]

m4_data = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive)



df_new.sort_values('Av_diff')

df_new.loc[:, rsel].mean()

ds.sel(bait=["CUL5", "ELOB", "CBFB", "LRR1"])['CRL_E'].mean(['rrep', 'preyu'])

nuts_kernal = numpyro.infer.NUTS(model2)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=50000, thinning=10)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, ds=ds, extra_fields=('potential_energy',))

df1

posterior_samples = mcmc.get_samples()

posterior_predictive = numpyro.infer.Predictive(model2, posterior_samples)(jax.random.PRNGKey(1), ds)

prior = numpyro.infer.Predictive(model2, num_samples=500)(jax.random.PRNGKey(2), ds)

numpyro_data = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive,
                              coords={"protein": ds.preyu.values, "cell": np.arange(5), "infection": np.arange(3)},
                              dims={"a": ["protein", "cell", "infection"]})

az.plot_trace(numpyro_data['sample_stats'], var_names=['lp'])

az.plot_trace(numpyro_data['posterior'], var_names=['Nc'])

post = numpyro_data['posterior']

az.plot_trace(post.sel(mu_dim_0=np.arange(0, 2)), var_names=['s'])

post

numpyro_data

numpyro_data['posterior']


def model3(ds, N=3062):
    
    # wt, vif, mock
    # 
    # [condition, bait, prey, rrep]
    
    ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_E'].values
    CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_E'].values
    CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_E'].values
    
    ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_E'].values
    CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_E'].values
    CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_E'].values
    
    ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_E'].values
    CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_E'].values
    CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_E'].values
    
    LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_E'].values
    
    ctrl_ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_C'].values
    
    ctrl_ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_C'].values
    
    ctrl_ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_C'].values
    
    ctrl_LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_C'].values
    
    
   # max_val = ds['CRL_E'].max('rrep')
    
   # mu_Nc = np.ones((5, 3))
   # mu_alpha = np.ones((N, 5, 3))
    

    
    #N = numpyro.sample('N', dist.Normal(np.zeros(3), 5))
    #mu = numpyro.sample('mu', dist.Normal(max_val.sel(bait='ELOB').values.T, 50), sample_shape=(3062, 3))
    #numpyro.sample('sc', dist.Normal(N * mu), obs=max_val.sel(bait='ELOB').values.T)
    
    
    
    
    #N1 = numpyro.sample('N1', dist.Normal(0, 1))
    #N2 = numpyro.sample('N2', dist.Normal(0, 1))
    
    #mu_elob = numpyro.sample('mu_elob', dist.Normal(np.mean(ELOB_wt, axis=1), np.var(ELOB_wt, axis=1)))
    #mu_cul5 = numpyro.sample('mu_cul5', dist.Normal(np.mean(CUL5_wt, axis=1), np.var(ELOB_wt, axis=1)))
    
    #numpyro.sample('ELOB_wt', dist.Normal(mu_elob * N1, 5), obs=ELOB_wt)
    #numpyro.sample('CUL5_wt', dist.Normal(mu_cul5 * N2, 5), obs=CUL5_wt)
    
    
    #cell_abundance = numpyro.sample(dist.Normal(jnp.ones((3, 5))), 1)
    
    assert ELOB_wt.shape == (3062, 4)
    
    mu_hyper_prior = np.ones((3062, 1)) / 50
    sig_hyper_prior = np.ones((3062, 1)) / 2
    
    
    mu = numpyro.sample('mu', dist.Exponential(mu_hyper_prior))
    sigma = numpyro.sample('s', dist.Exponential(sig_hyper_prior))
    
    Ncells = numpyro.sample('Nc', dist.Normal(np.ones((1, 4)), 0.5))
    
    Ncells_rep = jnp.repeat(Ncells, 3062, axis=0)
    
    
    numpyro.sample('sc', dist.Normal(mu * Ncells_rep, sigma), obs=ELOB_wt)
    
    #Ncells = cell_abundance * 1e7 
    
    #gamma_i = numpyro.sample('gamma', dist.Beta(0.5, 0.5), sample_shape=(3062,))
    #mu_ctrl = numpyro.sample('mu0', dist.Uniform(0, 250), sample_shape=(3062,))
    #mu_wt = numpyro.sample('mu_wt', dist.Uniform(0, 250), sample_shape=(3062,))
    
    #numpyro.sample('ELOB_wt', dist.Normal(mu_wt, 10), obs=ELOB_wt)
    #numpyro.sample('ctrl_ELOB_wt', dist.Normal(mu_ctrl * gamma_i, 10), obs=ctrl_ELOB_wt)

az.plot_forest(numpyro_data, var_names="N")

az.plot_trace(numpyro_data['posterior'], var_names=['N'])



a = jnp.ones((5,3)) * 2
b = jnp.arange(3062 * 5 * 3).reshape((3062, 5, 3))

(a * b)[:, :, 0]

samples = mcmc.get_samples()

import arviz as av


def condition_box_plot(ds, var=''):
    


ds.sel(preyu='ELOC')['CRL_C']

(ds.sel(bait=['ELOB', 'CBFB', 'CUL5'])['CRL_E'].sum('rrep') / 4 - ds['CRL_C'].sum('crep') / 12)

df_new.loc[:, 'log_odds_ratio'] = log_odds_ratio

df_new.loc[:, 'odds_ratio'] = odds_ratio

sns.scatterplot(df_new, x='log_odds_ratio', y='SaintScore')
plt.hlines(0.6, -3, 4, 'k')
plt.vlines(0, 0, 1, 'r')

# +
a = np.ravel(df_new.loc[:, [f"c{i}" for i in range(1, 5)]])
b = np.ravel(df_new.loc[:, [f"c{i}" for i in range(5, 9)]])
c = np.ravel(df_new.loc[:, [f"c{i}" for i in range(9, 13)]])
sns.boxplot([a, b, c])

sp.stats.ks_2samp(a, b)
# -

sp.stats.ks_2samp(a, a)

sp.stats.ks_2samp(b, c)

tmp = df_new['log_odds_ratio'].values.copy()
np.random.shuffle(tmp)
plt.plot(tmp, df_new['SaintScore'].values, 'k.')



 # ?sns.pairplot

n=600
df_new.sort_values('cAv', ascending=False).loc[:,
    ['rAv', 'cAv', 'rVar', 'cVar','bait', 'condition', 'SaintScore'] + rsel + csel ].iloc[n:50 + n]

query = 'PEBB_HUMAN'

df2[df2['PreyGene']==query]

df1[df1['PreyGene']==query]

df3[df3['PreyGene']==query]

tmp = np.ravel(ds.sel(bait='LRR1')['CRL_E'].values)
print(sum(tmp != 0))
plt.hist(tmp, bins=100)
plt.show()


# +
def m1(ds, preyrange=slice(0, None), 
       bait = ['CBFB', 'ELOB', 'CUL5'],
       conditions = ['wt', 'vif', 'mock']):
    
    
    prey_sel = ds.preyu[preyrange]
    
    d = ds.sel(bait=bait, condition=conditions, preyu=prey_sel, preyv=prey_sel)
    
    
    
# -

ds.sel(condition='mock', bait='CBFB')['CRL_E']

m1(ds)

ds.sel(bait=["CBFB", 'ELOB', 'CUL5'])

ds.sel(bait=['CBFB', 'ELOB', 'CUL5'])

ds.preyu[slice(0, None)]

ds.sel(bait=['LRR1', 'CBFB', 'CUL5', 'ELOB'], condition='mock')

df1[df1['Bait']=='CUL5wt_MG132'].sort_values('SaintScore', ascending=False).iloc[0:20, :]

bait_box_plot(ds, 'CRL_C')

ds.sel


def m2(ds):
    cbfb_preyname = 'Q13951'
    


ds.sel(condition='CBFB', preyu='CBFB')

df_new[df_new['bait'] == 'CBFB'].sort_values('SaintScore', ascending=False).iloc[0:20, :]

# +
# What values should we exclude from the analysis?

thresh_sel(0, log_odds_ratio)
# -

np.sum(odds_lambda == 0)

log_odds_ratio = np.log(odds_lambda) - np.log(odds_kappa)

plt.hist(np.array(odds_ratio), bins=100)
plt.show()

np.isnan(samples['k']).sum()

np.isnan(samples['l']).sum()

# +
# Vectorize loops
# -

a = np.array([[1, 2, 3],
              [5, 5, 5]])
b = np.array([10, 20])



jax.vmap(np.sum)(a, b.T)

jax.vmap(np.sum)(np.sum(a), b)



p1, p2 = posterior_odds(samples, df_new)

jax.scipy.stats.poisson.pmf(df_new[rsel].iloc[:, 0].values, 1)

jax.scipy.stats.poisson.pmf()

plt.hist(np.ravel(samples['l']), bins=100)
plt.show()

plt.hist(np.ravel(samples['k']), bins=100)
plt.show()





# +
def model(Yexp=None, Yctrl=None):
    
    numpyro.sample("C", dist.Poisson(kappa_), C=Yctrl)
    numpyro.sample("E", dist.Poisson(lambda_), E=Yexp)
    
    
    
# -

key = jax.random.PRNGKey(13)
ppc = jax.random.exponential(key)

# ?jax.random.exponential

df_new_all_json = df_new2json(df_new)
import json
with open("../sm1/df_new_all.json", 'w') as f:
    json.dump(df_new_all_json, f)

# + language="bash"
# ls ../sm1
# -

# ?json.dump

df_new_all_json



sp.stats.beta(0.1, 0.9).pdf(xaxis / 401)

(df_new['condition'] == 'wt') | (df_new['condition'] == 'mock')

chain_mapping[chain_mapping['PDBID'] == '4n9f']

prey2seq

df_newjson = df_new2json(df_new)

df_newjson['Ncsel']



hist_bins, bin_edges, patches = plt.hist(df_new['Av_diff'].values, bins=100, range=(-5, 5))
plt.vlines([-1, 1], 0, 2000, 'r')
plt.title
plt.show()

diff_sel = ~(np.abs(df_new['Av_diff'].values) <= 1)
sum(diff_sel)
df_new2json(df_new[diff_sel])

df_new[df_new['condition'] == 'vif'].loc['RUNX1', ['condition', 'bait'] + csel]

set(df_new['condition'])

df_new[df_new['Prey'] == 'P69723'].loc[:, ['bait', 'condition'] + csel]

# +
xaxis = np.arange(0, 10, 0.5)

nremaining = [sum(~(np.abs(df_new['Av_diff'].values) <= i)) for i in xaxis]
thresh = 1

plt.title("Filtering thresholds")
plt.plot(xaxis, nremaining, 'k.')
plt.vlines(1, 0, 21000, 'r')
plt.ylabel('N datapoints remaining')
plt.xlabel("Spectral count threshold")
plt.show()
# -

diff_sel = ~(np.abs(df_new['Av_diff'].values) <= 1)
sum(diff_sel)




# +
x, y = xy_from(prey_pairs_df, 'rand', np.arange(0, 0.05, 0.005))

npairs = len(prey_pairs_df)
npdb_pos = sum(prey_pairs_df['pdb_pos'].values)

yplot = y / npdb_pos
xplot = x / npairs
plt.plot(xplot, yplot, 'k.', label='Random Classifier')
plt.ylabel(f"Fraction PDB Positives (N={h(npdb_pos)})")
plt.xlabel(f"Fraction Total Positives (N={h(npairs)})")

y2 = 1.0
x2 = npdb_pos / npairs
plt.vlines([0.0035, 0.015], 0, 1, label="Estimated fraction of true PPIs")


plt.plot(x2, y2, 'r+', label='PDB Benchmark')
xmul = 10
plt.plot(x2 * xmul, y2, 'rx', label=f'{xmul}x PDB')
plt.savefig('f1.png', dpi=300)
plt.legend()
plt.show()
