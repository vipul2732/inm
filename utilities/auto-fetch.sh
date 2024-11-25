#!/bin/bash
source ~/.bashrc
if [ -z "$USER_REMOTE" ]; then
	echo "USER_REMOTE is not set. Exiting"
	exit 1
fi
MODEL1=mini_model23_n_mock_10k  
MODEL2=mini_model23_n_all_20k
MODEL3=mini_model23_n_all_2k

DEST=../results/temp

scp "$USER_REMOTE":~/Projects/inm/results/"$MODEL1"_merged/figs300.tar.gz "$DEST"/"$MODEL1"_merged/
scp "$USER_REMOTE":~/Projects/inm/results/"$MODEL2"_merged/figs300.tar.gz "$DEST"/"$MODEL2"_merged/
scp "$USER_REMOTE":~/Projects/inm/results/"$MODEL3"_merged/figs300.tar.gz "$DEST"/"$MODEL3"_merged/

for dname in $(ssh ajikarunia@dt2.wynton.ucsf.edu "ls ~/Projects/inm/results/ | grep mini")
do
	mkdir $DEST/$dname
    scp "$USER_REMOTE":~/Projects/inm/results/$dname/figs300.tar.gz "$DEST"/"$dname"/
    echo "Extracting $DEST/$dname/figs300.tar.gz"
    tar -xvf "$DEST"/"$dname"/figs300.tar.gz > /dev/null 2>&1 &
    scp "$USER_REMOTE":~/Projects/inm/results/$dname/representative_modeling_run.mp4 "$DEST"/"$dname"/
    scp "$USER_REMOTE":~/Projects/inm/results/$dname/representative_modeling_run_w_warmup.mp4 "$DEST"/"$dname"/
done
