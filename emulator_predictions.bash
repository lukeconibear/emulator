#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l h_vmem=4G

# Go to where you want to make the shortcut
cd /nobackup/earlacoa/machinelearning/temp

# Make a reference to the local disk location
ln -sf $TMPDIR datadir

# Copy the data you want from wherever into the local SSD space
rsync -a /nobackup/earlacoa/machinelearning/data/o3_emulators $TMPDIR/

# activate conda
conda activate pangeo_latest

# run python script
# ... can still now go to the localhost:5757 through the pangeo connection to see the progress
python /nobackup/earlacoa/dask/scripts/emulator_predictions.py /nobackup/earlacoa/machinelearning/temp/datadir/o3_emulators /nobackup/earlacoa/machinelearning/temp/datadir/

# move files back from local disk location
mv $TMPDIR/ds* /nobackup/earlacoa/machinelearning/data/summary/

