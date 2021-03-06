#!/bin/bash

GPU_ID=$1
SEED=$2
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --approach CSUR --alpha 0.1 --beta 0.1 --gamma 0.03 --logname $SEED'_CSUR' --seed $SEED
