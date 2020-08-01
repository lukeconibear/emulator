#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=00:05:00
#$ -pe smp 4
#$ -l h_vmem=4G

conda activate pangeo_latest
python custom_outputs_regrid.py
