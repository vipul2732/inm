#!/bin/bash
while true; do
	git pull origin main
	qsub qsub_run_tool_high_mem.sh mini_model23_n_mock_10k  
	qsub qsub_run_tool_high_mem.sh mini_model23_n_all_20k
	sleep 300
done
