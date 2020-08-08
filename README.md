## Emulator
### Scripts
- Merge shapefiles into a single multi-polygon (`create_merged_shapefile.ipynb`).  
- Crop emissions data to China shapefile training (`emis_apply_mask_train.py`) and test (`emis_apply_mask_test.py`) data.  
- Design of Latin hypercube designs (`latin_hypercube.ipynb`).  
- Output of Latin hypercube designs for training (`latin_hypercube_inputs_training.csv`) and test (`latin_hypercube_inputs_test.csv`) data.  
- Automatic machine learning tool (TPOT) using genetic programming to optimise model pipeline on 50 random grid cells (`tpot_optimiser.py`), with outputs in the `tpot_gridcells` folder (`tpot_emulator_pipeline_PM2_5_DRY_*.py`).  
- Configuration for TPOT based on Gaussian process regresor (`config_regressor_gaussian.py`).  
- Emulator cross-validation and sensitivity analysis (`emulator.ipynb`). Interactively computed on a HPC using Dask and Jupyter Lab following instructions [here](https://pangeo.io/setup_guides/hpc.html#).  
- Emulator predictions for custom inputs (`emulator_predictions.py`). Submitted to HPC (`emulator_predictions.bash`) using Dask for workers viewing worker status on Jupyter Lab. Can submit in batch mode (`emulator_predictions_batch.bash`).    
- Regrid custom outputs to population grid of the world (`custom_outputs_regrid.py`). Submitted to HPC (`custom_outputs_regrid.bash`) using Dask for workers viewing worker status on Jupyter Lab. Can submit in batch mode (`custom_outputs_regrid_batch.bash`).  
- Crop population-weighted output predictions to region's shapefile (`popweighted_region.py`). Submitted to HPC (`popweighted_region.bash`) using Dask for workers viewing worker status on Jupyter Lab. Uses cropping functions (`cutshapefile.py`).  
- Various emulator plots including emulator evaluation, sensitivity maps, prediction maps, and 2D contour pairs, (`emulator_plots.ipynb`).  

### Setup Python environment
- Create a conda environment with the required libraries from the config file (.yml) in the repository:
```
conda env create --name pangeo --file=pangeo.yml  
pip install salib dask_labextension pyarrow  
jupyter labextension install dask-labextension  
jupyter labextension install @jupyter-widgets/jupyterlab-manager  
```

### License  
This code is currently licensed under the GPLv3 License, free of charge for non-commercial use.  
[![DOI](https://zenodo.org/badge/249476351.svg)](https://zenodo.org/badge/latestdoi/249476351)  
