import pickle
import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import sklearn
from .regressor_gaussian import tpot_config

path = '/nobackup/earlacoa/machinelearning/data/'

with open(path + 'dfs_gridcell_01_t1-t18.pickle', 'rb') as ds:
    dfs_gridcell = pickle.load(ds)
    
dfs_combined = pd.concat(dfs_gridcell, ignore_index=True)

#targets = ['PM2_5_DRY', 'o3', 'AOD550_sfc', 'asoaX_2p5', 'bc_2p5', 'bsoaX_2p5', 'nh4_2p5', 'no3_2p5', 'oc_2p5', 'oin_2p5', 'so4_2p5']
target = 'PM2_5_DRY'

#np.random.seed(123)
#random_indexes = np.random.randint(0, len(dfs_gridcell.keys()), size=20)

#for target in targets:
X = dfs_combined[['RES', 'IND', 'TRA', 'AGR', 'POW']].values
y = dfs_combined[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

emulator = TPOTRegressor(generations=200, population_size=200, verbosity=2,
                         random_state=123, use_dask=True, n_jobs=-1, scoring='r2',
                         config_dict=tpot_config, subsample=0.001, cv=5)

emulator.fit(X_train, y_train)
print('training/validation 5-fold cv is the final one of the above')
print(f"test/holdout r2 = {emulator.score(X_test, y_test):.4f}")
emulator.export(path + 'tpot_emulator_pipeline_' + target + '_subsample0.001.py')
