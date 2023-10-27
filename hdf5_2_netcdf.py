#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: hdf5_2_netcdf.py
#------------------------------------------------------------------------------
# Version 0.2
# 27 October, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import netCDF4
import h5py
import os, glob

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

#variable = 'BA_Total'
#variable = 'BA_Forest_NonForest'
#variable = 'Cem_Total'
variable = 'Cem_Forest_NonForest'
            
latstep = 0.25
lonstep = 0.25
lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
    
#----------------------------------------------------------------------------
# METHODS:
#----------------------------------------------------------------------------

'''    
def array_to_xarray(lat, lon, da):

    da_xr = xr.DataArray( 
        da,
        dims = [ "latitude", "longitude",],
        coords = { "latitude": lat, "longitude": lon,},
        attrs = { "long_name": "data_per_pixel", "description": "data per pixel", "units": "",},
    )
        
    return da_xr
'''

def array_to_netcdf(lats, lons, da, time, variable, nc_file):

    #------------------------------------------------------------------------------
    # EXPORT: data to netCDF-4
    #------------------------------------------------------------------------------

    ds = xr.Dataset(
        {variable: (("lat", "lon"), da),
         },
        coords={
        "lat": ("lat", lats),
        "lon": ("lon", lons),
        },
    )
        
    # ADD: // global attributes
    
    ds.attrs['History'] = 'File generated on {} (UTC) by Michael Taylor'.format(datetime.utcnow().strftime('%c'))
    ds.attrs['Institution'] = 'Climatic Research Unit, University of East Anglia'
    ds.attrs['Licence'] = 'Data is licensed under the Open Government Licence v3.0 except where otherwise stated. To view this licence, visit https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3'
    ds.attrs['Reference'] = 'Jones, M. W., Abatzoglou, J. T., Veraverbeke, S., Andela, N., Lasslop, G., Forkel, M., ... & Le Quéré, C. (2022). Global and regional trends and drivers of fire under climate change. Reviews of Geophysics, 60(3), e2020RG000726.'
    ds.attrs['Source'] = 'Matt Jones'
    ds.attrs['Title'] = 'Global Wildfire Atlas data'
    ds.attrs['Version'] = 'wildfire-atlas-v0.1'
    ds.attrs['Conventions'] = 'CF-1.7'
    
    # ADD: time variable
    
    ds = ds.assign_coords(time = time)
    ds = ds.expand_dims(dim="time")                

    # SAVE:

    ds.to_netcdf( nc_file )    
    
#----------------------------------------------------------------------------
# LOAD: filelist
#----------------------------------------------------------------------------

a = glob.glob( 'DATA/' + variable + '/*.hdf5' )
filelist = sorted(a, reverse=False)
        
#----------------------------------------------------------------------------
# INITIALISE: xarray with time axis (replicates frame of zeros)
#----------------------------------------------------------------------------

# EXTRACT: datetime range from sorted filenames

year_start = filelist[0].split('/')[-1].split('.')[0][-6:-2]
year_end = filelist[-1].split('/')[-1].split('.')[0][-6:-2]
month_start = filelist[0].split('/')[-1].split('.')[0][-2:]
month_end = filelist[-1].split('/')[-1].split('.')[0][-2:]    
datetime_vec = pd.date_range( start = month_start + '-' + year_start, end = month_end + '-' + year_end, freq='MS') 

#----------------------------------------------------------------------------
# APPEND: frame from each file to xarray
#----------------------------------------------------------------------------

print('loading HDF5 frames ...')

for i in range(len(filelist)):
    
    f = h5py.File( filelist[i], 'r+' )
    keys = [key for key in f.keys()]
    #print(keys)
    key = keys[0]

    #subkeys = array(['TC10_F', 'TC10_NF', 'TC20_F', 'TC20_NF', 'TC30_F', 'TC30_NF',
    #   'TC40_F', 'TC40_NF', 'TC50_F', 'TC50_NF', 'TC60_F', 'TC60_NF',
    #   'TC70_F', 'TC70_NF', 'TC80_F', 'TC80_NF', 'TC90_F', 'TC90_NF'],
    #  dtype='<U7')

    if variable == 'BA_Total':
        da = np.array( f[key] )
    elif variable == 'BA_Forest_NonForest':
        subkeys = np.array( f[key] )                
        da = np.array( f[key][subkeys[4]] ) #TC30_F                
    elif variable == 'Cem_Total':
        da = np.array( f[key] )
    elif variable == 'Cem_Forest_NonForest':
        subkeys = np.array( f[key] )                
        da = np.array( f[key][subkeys[4]] ) #TC30_F                

    print(da.sum())
    #da[da == 0.0] = np.nan      

    #ds = array_to_xarray( lats, lons, da).expand_dims( time = datetime_vec )

    # SAVE: as netCDF4

    nc_file = 'OUT/' + filelist[i].split('/')[-1].split('.')[0] + '.nc'
    time = datetime_vec[i]
    array_to_netcdf(lats, lons, da, time, variable, nc_file)   
    
# -----------------------------------------------------------------------------
print('** END')
