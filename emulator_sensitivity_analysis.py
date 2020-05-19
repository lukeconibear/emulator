#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import xarray as xr
from SALib.sample import saltelli
from SALib.analyze import sobol
import joblib
import glob
import re

# setup emulator inputs and outputs

output = 'PM2_5_DRY' # should really parallelise over outputs
#output = 'o3'
#output = 'AOD550_sfc'
#output = 'asoaX_2p5'
#output = 'bc_2p5'
#output = 'bsoaX_2p5'
#output = 'nh4_2p5'
#output = 'no3_2p5'
#output = 'oc_2p5'
#output = 'oin_2p5'
#output = 'so4_2p5'

path = '~/' # change to data path

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

emulator_files = glob.glob(path + output + '/emulator_' + output + '_*.joblib')

for emulator_file in emulator_files: # should really parallelise over these though couldn't get working
    lat, lon = [float(item) for item in re.findall(r'\d+\.\d+', emulator_file)]
    emulator = joblib.load(emulator_file)

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


ds_sens_ind.to_netcdf(path + output + '/ds_sens_ind_' + output + '.nc')

