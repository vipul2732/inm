#!/bin/bash
# Fetch the significant PDBs and populate the significant cifs directory

start=${1:-1}
stop=${2:-50000}

for pdb_id in $(cat hhblits_out/SignificantPDB70_PDBIDs.csv | sed -n "$start,${stop}p"); do
    lower_pdb=${pdb_id,,}

    if [[ "$lower_pdb" == "3jaq" ]]; then
       echo "Obsolete PDB ID"
       echo "Remapping to 6GSM"
       lower_pdb="6gsm"
    elif [[ "$lower_pdb" == "6t8j" ]]; then
       echo "Obsolete PDB ID"
       echo "Remapping to 8BXX"
       lower_pdb="8bxx"
    elif [[ "$lower_pdb" == "4fxg" ]]; then
       echo "Obsolete PDB ID"
       echo "Remapping to 5jpm"
       lower_pdb="5jpm"
    elif [[ "$lower_pdb" == "3unr" ]]; then
       echo "Obsolete PDB ID"
       echo "Remapping to 4yta"
       lower_pdb="4yta"
    elif [[ "$lower_pdb" == "6emd" ]]; then
       echo "Obsolete PDB ID"
       echo "Remapping to 6i2d"
       lower_pdb="6i2d"
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
