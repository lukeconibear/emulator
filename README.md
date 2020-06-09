## Emulator
### Scripts
- Crop emissions data to China shapefile training (`emis_apply_mask_train.py`) and test (`emis_apply_mask_test.py`) data.  
- Design of Latin hypercube designs (`latin_hypercube.ipynb`).  
- Output of Latin hypercube designs for training (`latin_hypercube_inputs_training.csv`) and test (`latin_hypercube_inputs_test.csv`) data.  
- Automatic machine learning tool (TPOT) using genetic programming to optimise model pipeline on 50 random grid cells (`tpot_optimiser.py`), with outputs in the `tpot_gridcells` folder (`tpot_emulator_pipeline_PM2_5_DRY_*.py`).  
- Configuration for TPOT based on Gaussian process regresor (`config_regressor_gaussian.py`).  
- Emulator cross-validation and sensitivity analysis interactively computed on a HPC using Dask and Jupyter Lab (`emulator.ipynb`).  

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
