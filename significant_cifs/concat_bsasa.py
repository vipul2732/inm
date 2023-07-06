import pandas as pd
from pathlib import Path
from itertools import combinations
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.align
import sys
import numpy as np

bsasa = [pd.read_csv(str(i)) for i in Path(".").iterdir() if (i.suffix == (".csv") and ("bsasa_" in i.name))]
bsasa = pd.concat(bsasa)
bsasa.to_csv("BSASA_concat.csv", index=False)

