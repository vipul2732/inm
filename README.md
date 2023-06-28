## AP-MS PPI Benchmark 

Prey were taken from the authors exactly.
The following prey were excluded from the analysis

- IGHG1\_MOUSE


Viral proteins were mapped to the uniprot identifier with
the longest sequence by keyword search.

- vifprotein          :   P69723 
- polpolyprotein      :   Q2A7R5 
- nefprotein          :   P18801
- tatprotein          :   P0C1K3
- gagpolyprotein      :   P12493
- revprotein          :   P69718
- envpolyprotein      :   O12164


Table 1. Prey Table
PreyGene, UniprotId, UniprotSeq, UniprotSeqLen 

Uniprot sequences were fetched for all prey

HHBlits was used to search the PDB70 with default parameters. 

Table2.
eueryUID,PDB70ID,Probab,E-value,Score,Aligned\_cols,Identities,Similarity,Sum\_probs,Template\_Neff,Q,T 

Filter the pairwise alignments by resulting in 
- evalue > 1e-7
- aln\_len > 88
- %seq ID >= 30$ 

XX Alignments
