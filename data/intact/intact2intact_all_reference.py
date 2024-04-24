import pandas as pd

df = pd.read_csv("./intact.txt", sep="\t")

al = []
bl = []
for i, r in df.iterrows():
    a = r["#ID(s) interactor A"]
    b = r["ID(s) interactor B"]
    if ("uniprotkb" in a) and ("uniprotkb" in b):
        a = a.split(":")
        assert len(a) == 2
        b = b.split(":")
        assert len(b) == 2
        al.append(a[1])
        bl.append(b[1])
d = pd.DataFrame({"auid": al, "buid": bl})

d.to_csv("../processed/references/intact/intact_all_first_uid.tsv")
