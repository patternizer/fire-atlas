#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: convert_netcdf_SN_2_NS.py
#------------------------------------------------------------------------------
# Version 0.1
# 30 October, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------
# NB: MODIS --> HDF5 has latitudes running [-90,+90] top to bottom in the array so flip them!

import xarray as xr

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

variables = list([ 'BA_Total', 'BA_Forest_NonForest', 'Cem_Total', 'Cem_Forest_NonForest' ])

#----------------------------------------------------------------------------
# LOAD: loop over netCDF4 files, fix SN --> NS latitude issue
#----------------------------------------------------------------------------

for i in range(len(variables)):

    print( variables[i] )
    
    nc_file_in = 'OUT/netcdfs_SN_lat_order/' + variables[i] + '.nc'
    nc_file_out = 'OUT/' + variables[i] + '.nc'
    ds_in = xr.load_dataset( nc_file_in )
    ds = ds_in.copy()
    ds = ds.assign(variables={variables[i]:ds[variables[i]][::-1]})
    ds = ds.assign_coords({"lat": ds.lat * -1})
    ds.to_netcdf(  nc_file_out )
    
#----------------------------------------------------------------------------
print('** END')
