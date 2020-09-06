#!/bin/bash

GPU_ID=$1

for SEED in 1 2 3 ; do
 CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --approach CSAR --alpha 0.1 --beta 0.1 --gamma 0.03 --logname $SEED'_1' --seed $SEED
done
