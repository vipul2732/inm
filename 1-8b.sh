for pdbid in $(cat hhblits_out/nonself_pair_pdbs.csv); do
    pdb=${pdbid,,}
    cp all_bioassembly_cif/mmCIF/"${pdb:1:2}/$pdb.cif.gz" significant_cifs/
    gunzip -d significant_cifs/$pdb.cif.gz
done
