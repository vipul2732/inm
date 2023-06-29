#!/bin/bash

NFASTA=$(ls input_sequences/*.fasta | wc -l)
NHHR=$(ls hhblits_out/*.hhr | wc -l)
NALIGNMENTS=$(cat hhblits_out/PreyPDB70PairAlign.csv | wc -l)
NSIGALI=$(cat hhblits_out/SignificantPreyPDB70PairAlign.csv | wc -l)
NSELFPREYPAIR=$(cat hhblits_out/SignificantSelfPairs.csv | wc -l) 
NNONSELFPREYPAIR=$(cat hhblits_out/SignificantNonSelfPairs.csv | wc -l)

NPDBFILES=$(ls pdb_files/*.pdb | wc -l)
NCIFFILES=$(ls big_pdb_files/*.cif | wc -l)
NRENUPDB=$(ls renumbered_pdb_files/*.pdb | wc -l)
NRENUCIF=$(ls renumbered_big_pdb_files/*.cif | wc -l)
NRENMMTF=$(ls renumbered_mmtf/*.mmtf | wc -l)

echo "N-Fasta,$NFASTA"
echo "N-hhr,$NHHR"
echo "N-ali,$NALIGNMENTS"
echo "N-sig-ali,$NSIGALI"
echo "N-self-prey-pair,$NSELFPREYPAIR"
echo "N-nonself-prey-pair,$NNONSELFPREYPAIR"
echo "N-pdb,$NPDBFILES"
echo "N-cif,$NCIFFILES"
echo "N-repdb,$NRENUPDB"
echo "N-renucif,$NRENUCIF"
echo "N-renmmtf,$NRENMMTF"


