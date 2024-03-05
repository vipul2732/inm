import pandas as pd
from collections import defaultdict
import sys
sys.path.append("../../notebooks/")

import generate_benchmark_figures as gbf


d = pd.read_csv("HuRI.tsv", sep="\t", names=['a', 'b'])

a = set(d['a'].values)
b = set(d['b'].values)
a_b = a.union(b)

# Mapped IDs

id_map = pd.read_csv("idmapping_2024_02_24.tsv", sep="\t")
# Is id_map 1 to 1?

entry2ensemble_list = defaultdict(list)
for i, r in id_map.iterrows():
    entry2ensemble_list[r['Entry']].append(r['From'])

# 18 Uniprot Entries are supported by more than 1 ensemble ID
# Count an edge if either identifier pair counts it

a_b_idmap = set(id_map['From']).intersection(a_b) # 4361 ids
not_mapped = a_b - set(id_map['From'])

# Reviewed only - no
# id_map = id_map[id_map['Reviewed'] == 'reviewed']

a_b_idmap = set(id_map['From']).intersection(a_b) # 4340 ids
# Get the non self pairs 


ensemble2acc = {r['From'] : r['Entry'] for i,r in id_map.iterrows()}




d_edgelist = gbf.UndirectedEdgeList()
d_edgelist.update_from_df(d, a_colname="a", b_colname="b")

d_edgelist.reindex(ensemble2acc, enforce_coverage = False)




def dict_list_print(d):
    n = 0
    for key, l in d.items():
        if len(l) > 1:
            n += 1
            print(key, l)
    return n
