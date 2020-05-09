#!/usr/bin/env python3
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from tpot.export_utils import set_param_recursive

path = '/nobackup/earlacoa/machinelearning/data/'

with open(path + 'dict_train.pickle', 'rb') as ds:
    dict_train = pickle.load(ds)
    
with open(path + 'dict_test.pickle', 'rb') as ds:
    dict_test = pickle.load(ds)

df_train = pd.concat(dict_train, ignore_index=True)
df_test = pd.concat(dict_test, ignore_index=True)

inputs_train = pd.read_csv(path + 'latin_hypercube_inputs_train.csv')
inputs_test = pd.read_csv(path + 'latin_hypercube_inputs_test.csv')

df_eval = pd.DataFrame(columns=['output', 'y_test', 'y_pred', 'rmse_cv', 'r2_cv'])
df_eval_summary = pd.DataFrame(columns=['output', 'rmse_cv', 'r2_cv', 'rmse_test', 'r2_test', 'pearson_r2_test'])

lats = df_train[['lat', 'lon']].drop_duplicates()['lat'].values
lons = df_train[['lat', 'lon']].drop_duplicates()['lon'].values

outputs = ['PM2_5_DRY', 'o3', 'AOD550_sfc', 'asoaX_2p5', 'bc_2p5', 'bsoaX_2p5', 'nh4_2p5', 'no3_2p5', 'oc_2p5', 'oin_2p5', 'so4_2p5']

for output in outputs:
    for gridcell in df_train[['lat', 'lon']].drop_duplicates().values:
        lat, lon = gridcell
        df_train_gridcell = df_train.loc[df_train.lat == lat].loc[df_train.lon == lon]
        df_test_gridcell = df_test.loc[df_test.lat == lat].loc[df_test.lon == lon]
        
        X_train = inputs_train.values
        y_train = df_train_gridcell[output].values
        
        X_test = inputs_test.values
        y_test = df_test_gridcell[output].values

        emulator = make_pipeline(
            PowerTransformer(),
            GaussianProcessRegressor(kernel=Matern(length_scale=3.4000000000000004, nu=2.5), n_restarts_optimizer=240, normalize_y=False)
        )
        
        set_param_recursive(emulator.steps, 'random_state', 123)

        r2_cv = cross_val_score(emulator, X_train, y_train, cv=10, scoring='r2')
        rmse_cv = np.sqrt(np.abs(cross_val_score(emulator, X_train, y_train, cv=10, scoring='neg_mean_squared_error')))

        emulator.fit(X_train, y_train)
        y_pred = emulator.predict(X_test)

        df_eval = df_eval.append([{
            'output': output,                         
            'y_test': y_test, 
            'y_pred': y_pred,          
            'rmse_cv': rmse_cv, 
            'r2_cv': r2_cv}],
            ignore_index=True,
            sort=False)

    df_eval.to_csv(path + 'df_eval_'+output+'.csv')

    pattern = r'([+-]?\d+.?\d+)'

    rmse_cv = np.mean(pd.to_numeric(df_eval.loc[df_eval.output == output]['rmse_cv'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel())
    r2_cv   = np.mean(pd.to_numeric(df_eval.loc[df_eval.output == output]['r2_cv'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel())

    y_test = pd.to_numeric(df_eval.loc[df_eval.output == output]['y_test'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel()
    y_pred = pd.to_numeric(df_eval.loc[df_eval.output == output]['y_pred'].astype(str).str.extractall(pattern).squeeze(), errors='coerce').values.ravel()

    rmse_test = np.round(np.sqrt(np.abs(mean_squared_error(y_test, y_pred))), decimals=4)
    r2_test = np.round(r2_score(y_test, y_pred), decimals=4)
    pearson_r2_test = pearsonr(y_test, y_pred)[0] ** 2

    df_eval_summary = df_eval_summary.append([{
        'output': output,                          
        'rmse_cv': rmse_cv, 
        'r2_cv': r2_cv,                                             
        'rmse_test': rmse_test, 
        'r2_test': r2_test, 
        'pearson_r2_test': pearson_r2_test}],              
        ignore_index=True, 
        sort=False)

    df_eval_summary.to_csv(path + 'df_eval_summary_'+output+'.csv')
