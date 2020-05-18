#!/usr/bin/env python3
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from tpot.export_utils import set_param_recursive
import xarray as xr
from SALib.sample import saltelli
from SALib.analyze import sobol
import joblib

# setup emulator inputs and outputs

output = 'PM2_5_DRY' # could paralleise over outputs
#output = 'o3'

path = '~/' # change this to where the data is

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

# setup sensitivity analysis
sims = ['RES', 'IND', 'TRA', 'AGR', 'ENE']
sens_inds_S1_ST = ['S1', 'S1_conf', 'ST', 'ST_conf']
ds_sens_ind = xr.Dataset({})

empty_values = np.empty((580, 1440))
empty_values[:] = np.nan
empty_da = xr.DataArray(empty_values, dims=('lat', 'lon'), coords={'lat': np.arange(-60, 85, 0.25), 'lon': np.arange(-180, 180, 0.25)})

for sim in sims:
    for sens_ind in sens_inds_S1_ST:
        ds_sens_ind.update({sens_ind + '_' + sim: empty_da})
        
sims_S2 = ['RES_IND', 'RES_TRA', 'RES_AGR', 'RES_ENE', 'IND_TRA', 'IND_AGR', 'IND_ENE', 'TRA_AGR', 'TRA_ENE', 'AGR_ENE']
sims_S2_indexes = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
sens_inds_S2 = ['S2', 'S2_conf']

for sim in sims_S2:
    for sens_ind in sens_inds_S2:
        ds_sens_ind.update({sens_ind + '_' + sim: empty_da})

sens_inputs = {
    'num_vars': 5,
    'names': sims,
    'bounds': [[0.0, 1.5],
               [0.0, 1.5],
               [0.0, 1.5],
               [0.0, 1.5],
               [0.0, 1.5]]
}

sens_param_values = saltelli.sample(sens_inputs, 1000)

df_eval = pd.DataFrame(columns=['output', 'y_test', 'y_pred', 'rmse_cv', 'r2_cv'])
df_eval_summary = pd.DataFrame(columns=['output', 'rmse_cv', 'r2_cv', 'rmse_test', 'r2_test'])

for gridcell in df_train[['lat', 'lon']].drop_duplicates().values.tolist(): # should really parallelise over these (e.g. dask dataframe) though couldn't get working         
    lat, lon = gridcell
    df_train_gridcell = df_train.loc[df_train.lat == lat].loc[df_train.lon == lon]
    df_test_gridcell = df_test.loc[df_test.lat == lat].loc[df_test.lon == lon]
    
    X_train, X_test = inputs_train.values, inputs_test.values
    y_train, y_test = df_train_gridcell[output].values, df_test_gridcell[output].values

    emulator = make_pipeline(
        PowerTransformer(),
        GaussianProcessRegressor(kernel=Matern(length_scale=3.4000000000000004, nu=2.5), n_restarts_optimizer=240, normalize_y=False)
    )
    
    set_param_recursive(emulator.steps, 'random_state', 123)

    cv = cross_validate(emulator, X_train, y_train, cv=10, scoring={'r2': 'r2', 'rmse': 'neg_mean_squared_error'})
    emulator.fit(X_train, y_train)
    
    # can parallelise cross_validation and emulator fitting (if running interactively) with
    #with joblib.parallel_backend('dask'):    
    #    cv = cross_validate(emulator, X_train, y_train, cv=10, scoring={'r2': 'r2', 'rmse': 'neg_mean_squared_error'})
    #    emulator.fit(X_train, y_train)
    
    joblib.dump(emulator, path + output + '/emulator_' + output + '_' + str(lat) + '_' + str(lon) + '.joblib')

    r2_cv = cv['test_r2']
    rmse_cv = np.sqrt(np.abs(cv['test_rmse']))
    y_pred = emulator.predict(X_test)

    df_eval = df_eval.append([{
        'output': output,                          
        'y_test': y_test, 
        'y_pred': y_pred,                                             
        'rmse_cv': rmse_cv, 
        'r2_cv': r2_cv}],              
        ignore_index=True, 
        sort=False)

    sens_predictions = emulator.predict(sens_param_values)
    sens_ind_dict = sobol.analyze(sens_inputs, sens_predictions)

    for sim_index, sim in enumerate(sims):
        for sens_ind_index, sens_ind in enumerate(sens_inds_S1_ST):
            ds_sens_ind[sens_ind + '_' + sim] = xr.where(
               (ds_sens_ind.coords['lat'] == lat) & (ds_sens_ind.coords['lon'] == lon),
                sens_ind_dict[sens_ind][sim_index],
                ds_sens_ind[sens_ind + '_' + sim]
            )

    for sim_index, sim in enumerate(sims_S2):
        for sens_ind_index, sens_ind in enumerate(sens_inds_S2):
            ds_sens_ind[sens_ind + '_' + sim] = xr.where(
                (ds_sens_ind.coords['lat'] == lat) & (ds_sens_ind.coords['lon'] == lon),
               sens_ind_dict[sens_ind][sims_S2_indexes[sim_index][0], sims_S2_indexes[sim_index][1]],
                ds_sens_ind[sens_ind + '_' + sim]
            )

pattern = r'([+-]?\d+.?\d+)'

rmse_cv = np.mean(pd.to_numeric(df_eval.loc[df_eval.output == output]['rmse_cv'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel())
r2_cv   = np.mean(pd.to_numeric(df_eval.loc[df_eval.output == output]['r2_cv'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel())

y_test = pd.to_numeric(df_eval.loc[df_eval.output == output]['y_test'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel()
y_pred = pd.to_numeric(df_eval.loc[df_eval.output == output]['y_pred'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel()

rmse_test = np.round(np.sqrt(np.abs(mean_squared_error(y_test, y_pred))), decimals=4)
r2_test = np.round(r2_score(y_test, y_pred), decimals=4)

df_eval_summary = df_eval_summary.append([{
    'output': output,                          
    'rmse_cv': rmse_cv, 
    'r2_cv': r2_cv,                                             
    'rmse_test': rmse_test, 
    'r2_test': r2_test}],              
    ignore_index=True, 
    sort=False)

df_eval.to_csv(path + output + '/df_eval_' + output + '.csv')
ds_sens_ind.to_netcdf(path + output + '/ds_sens_ind_' + output + '.nc')
df_eval_summary.to_csv(path + output + '/df_eval_summary_' + output + '.csv')

