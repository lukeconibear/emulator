#!/usr/bin/env python3
import pickle
import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import sklearn
from config_regressor_gaussian import tpot_config

path = '/nobackup/earlacoa/machinelearning/data/'

with open(path + 'dict_train.pickle', 'rb') as ds:
    dict_train = pickle.load(ds)
    
with open(path + 'dict_test.pickle', 'rb') as ds:
    dict_test = pickle.load(ds)

df_train = pd.concat(dict_train, ignore_index=True)
df_test = pd.concat(dict_test, ignore_index=True)

inputs_train = pd.read_csv(path + 'latin_hypercube_inputs_train.csv')
inputs_test = pd.read_csv(path + 'latin_hypercube_inputs_test.csv')

lats = df_train[['lat', 'lon']].drop_duplicates()['lat'].values
lons = df_train[['lat', 'lon']].drop_duplicates()['lon'].values

output = 'PM2_5_DRY'

np.random.seed(123)
random_indexes = np.random.randint(low=0, high=np.shape(df_train[['lat', 'lon']].drop_duplicates().values)[0], size=50)

for index, random_index in enumerate(random_indexes):
    lat, lon = df_train[['lat', 'lon']].drop_duplicates().values[random_index]

    df_train_gridcell = df_train.loc[df_train.lat == lat].loc[df_train.lon == lon]
    df_test_gridcell = df_test.loc[df_test.lat == lat].loc[df_test.lon == lon]
    
    X_train = inputs_train.values
    y_train = df_train_gridcell[output].values
    
    X_test = inputs_test.values
    y_test = df_test_gridcell[output].values
    
    emulator = TPOTRegressor(
        generations=5,
        population_size=50,
        verbosity=2,
        random_state=123,
        use_dask=True,
        n_jobs=-1,
        scoring='r2',
        config_dict=tpot_config,
        cv=10
    )

    emulator.fit(X_train, y_train)
    print('training/validation 10-fold cv is the final one of the above')
    print(f"test/holdout r2 = {emulator.score(X_test, y_test):.4f}")
    emulator.export(path + 'tpot_emulator_pipeline_' + output + '_' + str(lat) + '_' + str(lon) + '.py')

