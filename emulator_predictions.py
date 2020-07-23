#!/usr/bin/env python3
import glob
import os
import re
import sys
import time
import dask.bag as db
from dask_jobqueue import SGECluster
from dask.distributed import Client
import joblib
import numpy as np
import pandas as pd
import xarray as xr

# global variables
output = 'o3'
data_dir = sys.argv[1]
out_dir = sys.argv[2]
EMULATORS = None

def get_emulator_files(file_path=data_dir, file_pattern='emulator*'):
    emulator_files = glob.glob(os.sep.join([file_path, file_pattern]))
    return emulator_files

def load_emulator(emulator_file):
    lat, lon = [float(item) for item in re.findall(r'\d+\.\d+', emulator_file)]
    emulator = joblib.load(emulator_file)
    return lat, lon, emulator

def create_dataset(results):

    res = results[0]['res']
    ind = results[0]['ind']
    tra = results[0]['tra']
    agr = results[0]['agr']
    ene = results[0]['ene']
    filename = f'RES{res:.1f}_IND{ind:.1f}_TRA{tra:.1f}_AGR{agr:.1f}_ENE{ene:.1f}'

    lat = [item['lat'] for item in results]
    lon = [item['lon'] for item in results]
    result = [item['result'] for item in results]

    df_results = pd.DataFrame([
        lat,
        lon,
        result
    ]).T
    df_results.columns = ['lat', 'lon', output]
    df_results = df_results.set_index(['lat', 'lon']).sort_index()
    ds_custom_output = xr.Dataset.from_dataframe(df_results)
    ds_custom_output.to_netcdf(
        out_dir + 'ds_' + filename + '_' + output + '.nc'
    )

def custom_predicts(custom_input):

    def emulator_wrap(emulator):
        lat, lon, emulator = emulator
        return {
            'lat': lat,
            'lon': lon,
            'res': custom_input[0][0],
            'ind': custom_input[0][1],
            'tra': custom_input[0][2],
            'agr': custom_input[0][3],
            'ene': custom_input[0][4],
            'result': emulator.predict(custom_input)[0]
        }

    global EMULATORS
    if not EMULATORS:
        emulator_files = get_emulator_files()
        EMULATORS = list(map(load_emulator, emulator_files)) 
    emulators = EMULATORS

    results = list(map(emulator_wrap, emulators))
    create_dataset(results)

def main():
    # dask cluster and client
    n_processes = 1
    n_jobs = 30
    n_workers = n_processes * n_jobs
    n_memory = 1

    cluster = SGECluster(
        interface='ib0',
        walltime='01:00:00',
        memory=f'{n_processes * n_memory:.0f} G',
        resource_spec=f'h_vmem={n_memory:.0f}G',
        scheduler_options={
            'dashboard_address': ':5757',
        },
        job_extra = [
          '-cwd',
          '-V',
          f'-pe smp {n_processes}'
        ],
        local_directory = os.sep.join([
          os.environ.get('PWD'),
          'dask-worker-space'
        ])
    )

    client = Client(cluster)

    cluster.scale(jobs=n_jobs)

    time_start = time.time()

    # custom inputs
    matrix_stacked = np.array(np.meshgrid(
        np.linspace(0, 1.4, 8), # 1.5 and 16 for 0.1, 1.5 and 6 for 0.3, 1.4 and 8 for 0.2
        np.linspace(0, 1.4, 8),
        np.linspace(0, 1.4, 8),
        np.linspace(0, 1.4, 8),
        np.linspace(0, 1.4, 8)
    )).T.reshape(-1, 5)
    custom_inputs_set = set(tuple(map(float, map("{:.1f}".format, item))) for item in matrix_stacked)

    custom_inputs_completed_filenames = glob.glob('/nobackup/earlacoa/machinelearning/data/summary/ds*' + output + '*')
    custom_inputs_completed_list = []
    for custom_inputs_completed_filename in custom_inputs_completed_filenames:
        custom_inputs_completed_list.append(
            [float(item) for item in re.findall(r'\d+\.\d+', custom_inputs_completed_filename)]
        )
        
    custom_inputs_completed_set = set(tuple(item) for item in custom_inputs_completed_list)
    custom_inputs_remaining_set = custom_inputs_set - custom_inputs_completed_set
    custom_inputs = [np.array(item).reshape(1, -1) for item in custom_inputs_remaining_set]
    print(f'custom inputs remaining for {output}: {len(custom_inputs)}')

    # dask bag and process
    custom_inputs_process = 1000
    print(f'predicting for {custom_inputs_process} custom inputs ...')
    bag_custom_inputs = db.from_sequence(custom_inputs[0:custom_inputs_process], npartitions=n_workers)
    bag_custom_inputs.map(custom_predicts).compute()

    time_end = time.time() - time_start
    print(f'completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours')
    print(f'average time per custom input is {time_end / custom_inputs_process:0.2f} seconds')

    client.close()
    cluster.close()

if __name__ == '__main__':
    main()

