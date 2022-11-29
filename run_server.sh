#!/bin/bash

cd /mnt/shared/mittone/GraphDTA
#conda activate geometric

nohup python server.py --num-rounds 500 --num-clients 8 --folder data --seed $RANDOM --early-stop 50 --save-name "model_nonIID_fl_protein_8" --normalisation ln &> output_nonIID_fl_protein_8 &

