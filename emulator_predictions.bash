#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l h_vmem=4G

conda activate pangeo_latest

python emulator_predictions.py

# ... can still now go to the localhost:5757 through the pangeo connection to see the progress
