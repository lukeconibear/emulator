#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
import numpy as np
import salem
import xarray as xr
from rasterio import features
from affine import Affine

def transform_from_latlon(lat, lon):
    """ input 1D array of lat / lon and output an Affine transformation """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    
    return trans * scale


def rasterize(shapes, coords, latitude='latitude', longitude='longitude',
              fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

# these need to be a single multipolygon
shp_china = gpd.read_file('/nobackup/earlacoa/emissions/shapefiles/CHN_adm0.shp') # not in repo, can access online or supply on request
shapes_china = [(shape, n) for n, shape in enumerate(shp_china.geometry)]

sim_list = list(map(str, list(range(51, 56))))
sim_list = ['t' + sim for sim in sim_list]
test_data = pd.read_csv('latin_hypercube_inputs_test.csv', header=0).values

scenarios = ['emis_' + sim for sim in ['res', 'ind', 'tra', 'agr', 'ene']]

for sim in sim_list:
    emis_files = []
    emis_files.extend(glob.glob('/nobackup/earlacoa/emissions/EDGAR-HTAP2_MEIC2015/MOZART_' + sim + '/*.nc')) # not in repo, can access online or supply on request
    emis_files = [emis for emis in emis_files if 'CH4' not in emis]
    for emis_file in emis_files:
        for index, scenario in enumerate(scenarios):
            emis = salem.xr.open_dataset(emis_file) # open original, then open updated per scenario change (due to emis_file changing)
            if (scenario in emis) == True:
                emis_file = emis_file[:-3] + '_' + scenario + '.nc' # update emis_file filename to add the scenario to the end
                print('updating for: ', emis_file)
                sec = emis[scenario]
				# mark shapefiles with 1 or np.nan (needs the extra step)
				sec['china'] = rasterize(shapes_china, sec.coords, longitude='lon', latitude='lat') # in shapefile == 0, outside == np.nan
				sec['china'] = sec.china.where(cond=sec.china!=0, other=1) # if condition (outside china, as inside == 0) preserve, otherwise (1, to mark in china)
				# if condition is shapefile (==1) or not (!=1) preserve, otherwise replace with
				sec = sec.where(cond=sec.china!=1, other=sec * test_data[int(sim.replace('t', '')) - 50][index]) # if condition (not in china) preserve, otherwise (in china, and scale)
                emis[scenario] = sec
                print('writing updated: ', emis_file)
                emis.to_netcdf(emis_file)
                emis.close()
            else:
                emis.close()

