#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l h_vmem=4G
#$ -m ea
#$ -M earlacoa@leeds.ac.uk
#$ -P admiralty

# --- without $TMPDIR ---
conda activate pangeo_latest
python /nobackup/earlacoa/dask/scripts/emulator_predictions.py /nobackup/earlacoa/machinelearning/data/PM2_5_DRY_emulators /nobackup/earlacoa/machinelearning/data/summary/

# --- with $TMPDIR ---
# need to all be on the same node to work
# admiralty sometimes fulfills this requirement
# otherwise require 1 node from here: -l np=1
# and run all of the workers from a LocalCluster (not SGECluster): client = Client(n_workers=39)

# 1. go to where you want to make the shortcut
#cd /nobackup/earlacoa/machinelearning/temp

# 2. make a reference to the local disk location
#ln -sf $TMPDIR datadir

# 3. copy the data you want from wherever into the local SSD space
#rsync -a /nobackup/earlacoa/machinelearning/data/PM2_5_DRY_emulators $TMPDIR/

# 4. activate conda
#conda activate pangeo_latest

# 5. run python script (can still now go to the localhost:5757 through the pangeo connection to see the progress)
#python /nobackup/earlacoa/dask/scripts/emulator_predictions.py /nobackup/earlacoa/machinelearning/temp/datadir/PM2_5_DRY_emulators /nobackup/earlacoa/machinelearning/temp/datadir/

# 6. move files back from local disk location
#mv $TMPDIR/ds* /nobackup/earlacoa/machinelearning/data/summary/





