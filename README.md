# Emulator
## Scripts
- Maximim Latin hypercube space filling design for training data setup (latin_hypercube.ipynb).
- Test emulator of Gaussian process (GP) regressor for 20 random grid cells (emulator_gp-matern.ipynb).
- Randomised cross-validation of various GP kernels and their hyperparameters (randomised-cv_gp-kernels.ipynb).
- Bayesian cross-validation of hyperparameters for GP with Matern kernel (bayesian-cv_gp-matern.ipynb).
- Automatic machine learning tool using genetic programming to optimise model pipeline (optimise_pipeline_gp-matern.ipynb).

## Setup Python environment
- Create a conda environment with the required libraries from the config file (.yml) in the repository:
```
conda env create --name emulator --file=emulator.yml
```
- Or create your own using:
```
conda create -n emulator -c conda-forge numpy matplotlib pandas scipy seaborn scikit-learn tpot dask dask-ml scikit-optimize pydoe jupyter
```