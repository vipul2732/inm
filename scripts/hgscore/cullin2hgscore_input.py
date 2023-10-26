import sys

sys.path = ["../../notebooks/"] + sys.path

from _BSASA_functions import *
from _BSASA_imports import *
from types import SimpleNamespace

import pandas as pd
import pickle as pkl

auth_prey_gene_and_uid = SimpleNamespace()
auth_prey_gene_and_uid.df = pd.read_csv("../../table1.csv")
auth_prey_gene_and_uid.gene2uid = {
        key:val for key, val in auth_prey_gene_and_uid.df.values}
auth_prey_gene_and_uid.uid2gene = {
     val:key for key,val in auth_prey_gene_and_uid.gene2uid.items()}
cullin_data = SimpleNamespace()

df1, df2, df3 = init_dfs(xlsx_path="../../1-s2.0-S1931312819302537-mmc2.xlsx")
cullin_data.df1 = df1
cullin_data.df2 = df2
cullin_data.df2 = df3

cullin_data.df_all = init_df_all(df1, df2, df3, auth_prey_gene_and_uid.gene2uid)

cullin_data.df_all = update_df_all_bait_and_condition(cullin_data.df_all)

cullin_data.rsel = [f"r{i}" for i in range(1, 5)]
cullin_data.csel = [f"c{i}" for i in range(1, 13)]
cullin_data.df_all = parse_spec(cullin_data.df_all)

with open("../../notebooks/df_new.pkl", "rb") as f:
    df_new = pkl.load(f)

# Bait, Prey, Condition, Controls

baits = set(df_new['bait'])
rsel = [f"r{i}" for i in range(1, 5)]

idRun = []
idPrey = []
countPrey = []
lenPrey = []

# Prepare input data for HGSCore
for i, row in df_new.iterrows():
   bait = row['bait']
   condition = row['condition']
   prey = row['PreyName']
   aa_len = row['aa_seq_len']
   for j, replicate in enumerate(rsel):
       count = row[replicate]
       run_id = bait + "_" + condition + "_" + str(j) 
       idRun.append(run_id)
       idPrey.append(prey)
       countPrey.append(count)
       lenPrey.append(aa_len)

df = pd.DataFrame({"idRun": idRun,
    "idPrey": idPrey, "countPrey": countPrey, "lenPrey": lenPrey})

df.to_csv("cullin_hgscore_input.csv", index=False)


   


