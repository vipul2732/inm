#!/bin/bash
# Fetch the significant PDBs and populate the significant cifs directory

for pdb_id in $(cat hhblits_out/SignificantPDB70_PDBIDs.csv); do
    lower_pdb=${pdb_id,,}

    if [[ "$lower_pdb" == "3jaq" ]]; then
       echo "Obsolete PDB ID"
       echo "Remapping to 6GSM"
       lower_pdb="6gsm"
    fi
       
    middle=${lower_pdb:1:2}
    filename=all_bioassembly_cif/mmCIF/$middle/${lower_pdb}.cif.gz

    if [[ ! -f "$filename" ]]; then
        echo "Error: File does not exist" >&2
        echo "$filename"
        exit 1
    fi

    rsync -v --ignore-existing $filename significant_cifs/

    if [[ ! -f "${lower_pdb}.cif" ]]; then
        gunzip -c significant_cifs/${lower_pdb}.cif.gz > significant_cifs/${lower_pdb}.cif
    else
       echo "File significant_cifs/${lower_pdb}.cif already exists." >&2
    fi
done
