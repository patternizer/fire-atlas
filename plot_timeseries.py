#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_timeseries.py
#------------------------------------------------------------------------------
# Version 0.1
# 27 October, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
dpi = 300

#variable = 'BA_Total'
#variable = 'BA_Forest_NonForest'
#variable = 'Cem_Total'
variable = 'Cem_Forest_NonForest'

nc_file = 'OUT/' + variable + '.nc'

#----------------------------------------------------------------------------
# LOAD: netCDF4
#----------------------------------------------------------------------------

ds = xr.load_dataset( nc_file )

#----------------------------------------------------------------------------
# FIND: highest variance gridcell
#----------------------------------------------------------------------------

lats = ds.lat.values
lons = ds.lon.values

# COMPUTE: unbiased esimtate of SD per gridcell (along time axis)

sd = ds[variable].std(skipna=True, ddof=1, dim="time")

# FIND: gridcell containing max SD

sd_max = sd.where(sd==sd.max(), drop=True).squeeze()
lat_max = sd_max.lat.values + 0
lon_max = sd_max.lon.values + 0

# EXTRACT: timeseries

ts = ds[variable].sel(lon=lon_max, lat=lat_max, method="nearest")

#----------------------------------------------------------------------------
# PLOT
#----------------------------------------------------------------------------

figstr = variable + '.png'
titlestr = variable + ': highest variance gridcell (' + str(lat_max) + r'$^{\circ}$N' + ',' + str(lon_max) + r'$^{\circ}$E' + ')'
if (variable == 'BA_Forest_NonForest') | (variable == 'Cem_Forest_NonForest'):
    ystr = 'TC30_F'
else:
    ystr = 'Total'             
           
fig, ax = plt.subplots(figsize=(15,10))     
plt.plot( ds.time, ts, ls='-', lw=1, color='k')
plt.ylabel(ystr, fontsize=fontsize, color='k')
plt.tick_params(labelsize=fontsize, colors='k')    
#plt.legend(loc='upper left', fontsize=12)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#----------------------------------------------------------------------------
print('** END')
