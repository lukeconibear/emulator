#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -pe smp 1
#$ -l h_vmem=8G

conda activate pangeo_latest
python popweighted_region.py
