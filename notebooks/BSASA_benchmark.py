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

# +
chain_mapping = pd.read_csv("../significant_cifs/chain_mapping.csv")
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

sum(prey_pairs_df['pdb_pos'].values)

import operator

prey_pairs_df['rand'] < 2





# ?plt.legend

sum(prey_pairs_df['pdb_pos'])


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



cocomplex_matrix.loc[p1, p2]

np.array(tensorC).shape

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
# -

d = pd.read_excel("../1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=2)

sel = d['Bait'] == 'LRR1mock_MG132'
d[sel].sort_values('SpecSum')[d['PreyGene'] == 'LLR1_HUMAN']

# +
import operator
# Function for plotting an accuracy curve
def npos_ntotal(prey_pairs_df, col, threshold, comp=operator.le):
    sel = prey_pairs_df[col] <= threshold
    sub_df = prey_pairs_df.loc[sel, :]
    npos = np.sum(sub_df['pdb_pos'].values)
    ntotal = len(sub_df)
    return npos, ntotal

def xy_from(prey_pairs_df, col, thresholds):
    npos = []
    ntot = []
    for t in thresholds:
        p, nt = npos_ntotal(prey_pairs_df, col, t)
        npos.append(p)
        ntot.append(nt)
    return np.array(ntot), np.array(npos)

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
