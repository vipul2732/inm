#!/bin/bash
grep -f ../../results/illustration_wt_vif/CRL5_UIDS RF_SCORES > temp.txt
grep -f ../../results/illustration_wt_vif/CRL5_UIDS_UNDERSCORE temp.txt > CRL5_RF_SCORES


