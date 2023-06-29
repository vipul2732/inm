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
The following entries were obsolete.
- P30042  :  
- Q9Y2S0  :  

For these obsolete entries the most recent uniref sequence was saved instead

HHBlits was used to search the PDB70 with default parameters. 

Table2. Significant HHR Hit Table
QueryUID,PDB70ID,Probab,E-value,Score,Aligned\_cols,Identities,Similarity,Sum\_probs,Template\_Neff,Q,T 

Filter the pairwise alignments by resulting in 
- evalue <= 1e-7
- aln\_len > 88
- %seq ID >= 30% 

XX Significant Alignments

Two queries may map to the same chain in the template
or they may map to different chains (Self vs Non-self).

Table 3. Non Self Significant Pair Table
Query1, Query2, PDB70ID1, PDB70ID2 

Edge cases were handled

PDB Files were renumbered according to their input sequence
alignemnts.

BSASA was calculated for all pairs of renumbered pdb files 

Any Prey pair may the following relationships.
- Not found in the same PDB
- Found in one PDB
  - Map to a single chain
  - Map to multiple chains
 
- Found in multiple PDB's 

Table 4. PDB70ID, SASA
   
Table 5. PDB70ID1, PDB70ID2, SASA1, SASA2, BSASA, QUERIES1, QUERIES2 


