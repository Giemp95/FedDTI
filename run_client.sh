#!/bin/bash

cd /mnt/shared/mittone/GraphDTA
#source activate geometric

nohup python training_fl.py --server a40-node3:8080 --seed $RANDOM  --folder data --early-stop 50 --normalisation ln --num-clients 8 --partition $1 --diffusion --diffusion-folder "/run_nonIID_fl_protein_8" &

