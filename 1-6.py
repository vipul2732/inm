"""
Filter Out The Significant Hits
"""

import pandas as pd
df = pd.read_csv("hhblits_out/PreyPDB70PairAlign.csv")

df = df[df['E-value'] >= 1e-7]
df = df[df['Aln_cols'] >= 88]
df.loc[:, 'Identities'] = [int(i.strip("%") for i in df['Identities'].values]
df = df[df['Identities'] >= 30]
df.to_csv("SignificantPreyPDB70PairAlign.csv", index=False)
