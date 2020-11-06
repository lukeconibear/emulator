#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l h_vmem=32G

conda activate pangeo_latest
python regrid_to_popgrid.py
