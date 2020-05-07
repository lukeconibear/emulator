# Emulator
## Scripts
- Crop emissions data to China shapefile training (`emis_apply_mask_train.py`) and test (`emis_apply_mask_test.py`) data.  
- Design of Latin hypercube designs (`latin_hypercube.ipynb`).  
- Output of Latin hypercube designs for training (`latin_hypercube_inputs_training.csv`) and test (`latin_hypercube_inputs_test.csv`) data.  
- Automatic machine learning tool using genetic programming to optimise model pipeline on 50 random grid cells (`tpot_optimiser.py`).  
- Emulator run and evaluation (`emulator.py`).  

## Setup Python environment
- Create a conda environment with the required libraries from the config file (.yml) in the repository:
```
conda env create --name emulator --file=emulator.yml
```
- Or create your own using:
```
conda create -n emulator -c conda-forge numpy matplotlib pandas scipy seaborn scikit-learn tpot dask dask-ml scikit-optimize pydoe jupyter
```
