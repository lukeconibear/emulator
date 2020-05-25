## Emulator
### Scripts
- Crop emissions data to China shapefile training (`emis_apply_mask_train.py`) and test (`emis_apply_mask_test.py`) data.  
- Design of Latin hypercube designs (`latin_hypercube.ipynb`).  
- Output of Latin hypercube designs for training (`latin_hypercube_inputs_training.csv`) and test (`latin_hypercube_inputs_test.csv`) data.  
- Automatic machine learning tool (TPOT) using genetic programming to optimise model pipeline on 50 random grid cells (`tpot_optimiser.py`), with outputs in the `tpot_gridcells` folder (`tpot_emulator_pipeline_PM2_5_DRY_*.py`).  
- Configuration for TPOT based on Gaussian process regresor (`config_regressor_gaussian.py`).  
- Emulator cross-validation (`emulator_cross_validation.py`).  
- Emulator sensitivity analysis (`emulator_sensitivity_analysis.py`).  

### Setup Python environment
- Create a conda environment with the required libraries from the config file (.yml) in the repository:
```
conda env create --name emulator --file=emulator.yml  
pip install SALib  
```
- Or create your own using:
```
conda create -n emulator -c conda-forge numpy matplotlib pandas scipy seaborn scikit-learn tpot dask dask-ml scikit-optimize pydoe jupyter  
pip install SALib  
```  
### License  
This code is currently licensed under the GPLv3 License, free of charge for non-commercial use.  
[![DOI](https://zenodo.org/badge/249476351.svg)](https://zenodo.org/badge/latestdoi/249476351)  
