#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_ipcc_ar6_land_aggregated_stats.py
#------------------------------------------------------------------------------
# Version 0.3
# 1 November, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

# Dataframe libraries:

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

# Plotting libraries:
    
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# %matplotlib inline # for Jupyter Notebooks
import seaborn as sns; sns.set()

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
dpi = 300

use_log = False
use_smooth = False
use_abbrevs = True

variable = 'BA_Total'
#variable = 'BA_Forest_NonForest'
#variable = 'Cem_Total'
#variable = 'Cem_Forest_NonForest'

use_yearly = False
if use_yearly == True:
    temporalstr = 'yearly'

    nsmooth = 2 
    use_all_but_last_year = True
    
else:
    temporalstr = 'monthly'

    nsmooth = 12
    use_all_but_last_year = False

nc_file = 'OUT/' + variable + '.nc'

#----------------------------------------------------------------------------
# LOAD: netCDF4
#----------------------------------------------------------------------------

ds = xr.load_dataset( nc_file )

#----------------------------------------------------------------------------
# EXTRACT: regional mean timeseries and compute statistics
#----------------------------------------------------------------------------

if use_log == True:
    
    Z = ds[variable]
    Z = Z + 1.0
    Z = np.log10(Z)

else:
    
    Z = ds[variable]

if use_yearly == True:

    Z = Z.resample(time="AS").sum(dim="time")   

    if use_all_but_last_year == True:    
        
        Z = Z.sel(time=slice(Z.time[0],Z.time[-2]))

# SET: zeros to NaN

Z = Z.where(Z > 0.0)

mask_3D = regionmask.defined_regions.ar6.land.mask_3D( Z )
n_regions = mask_3D.region.shape[0]

#region_i = mask_3D.sel(region=3)
#Z_i = Z.where(region_i)

weights = np.cos(np.deg2rad(Z.lat))
Z_regional_weighted_mean = Z.weighted(mask_3D * weights).mean(dim=("lat", "lon"))
Z_regional_weighted_median = Z.weighted(mask_3D * weights).quantile(0.500, dim=("lat", "lon"))
Z_regional_weighted_250 = Z.weighted(mask_3D * weights).quantile(0.250, dim=("lat", "lon"))
Z_regional_weighted_750 = Z.weighted(mask_3D * weights).quantile(0.750, dim=("lat", "lon"))
Z_regional_weighted_100 = Z.weighted(mask_3D * weights).quantile(0.100, dim=("lat", "lon"))
Z_regional_weighted_900 = Z.weighted(mask_3D * weights).quantile(0.900, dim=("lat", "lon"))
Z_regional_weighted_025 = Z.weighted(mask_3D * weights).quantile(0.025, dim=("lat", "lon"))
Z_regional_weighted_975 = Z.weighted(mask_3D * weights).quantile(0.975, dim=("lat", "lon"))

#----------------------------------------------------------------------------
# PLOT: facetgrid of regionally-averaged timeseries gridcell stats
#----------------------------------------------------------------------------

figstr = variable + '-' + 'ipcc-ar6-land-region-timeseries' + '-' + 'stats' + '-' + temporalstr + '.png'
titlestr = variable + ': gridcell statistics across each IPCC AR6 land region (1-46)'

fig, axs = plt.subplots(ncols=8, nrows=6, figsize=(13.33,7.5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.3, 'wspace': 0.1})
axs = axs.ravel()
for i in range(n_regions):
    
    if use_smooth == True:
                
        axs[i].fill_between( Z.time.values, pd.Series( Z_regional_weighted_025.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), pd.Series( Z_regional_weighted_975.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), alpha=0.25, color='lightblue', label='95% CI' )
        axs[i].fill_between( Z.time.values, pd.Series( Z_regional_weighted_100.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), pd.Series( Z_regional_weighted_900.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), alpha=0.1, color='blue', label='90% CI' ) 
        axs[i].fill_between( Z.time.values, pd.Series( Z_regional_weighted_250.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), pd.Series( Z_regional_weighted_750.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), alpha=0.1, color='purple', label='IQR' ) 
        axs[i].plot( Z.time.values, pd.Series( Z_regional_weighted_median.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), color='purple', lw=0.5, alpha = 1, label='Median' )
        axs[i].plot( Z.time.values, pd.Series( Z_regional_weighted_mean.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), color='red', lw=0.5, alpha = 1, label='Mean' )

    else:

        axs[i].fill_between( Z.time.values, Z_regional_weighted_025.isel(region=i).values, Z_regional_weighted_975.isel(region=i).values, alpha=0.25, color='lightblue', label='95% CI' ) 
        axs[i].fill_between( Z.time.values, Z_regional_weighted_100.isel(region=i).values, Z_regional_weighted_900.isel(region=i).values, alpha=0.1, color='blue', label='90% CI' ) 
        axs[i].fill_between( Z.time.values, Z_regional_weighted_250.isel(region=i).values, Z_regional_weighted_750.isel(region=i).values, alpha=0.1, color='purple', label='IQR' ) 
        axs[i].plot( Z.time.values, Z_regional_weighted_median.isel(region=i).values, color='purple', lw=0.5, alpha = 1, label='Median' )
        axs[i].plot( Z.time.values, Z_regional_weighted_mean.isel(region=i).values, color='red', lw=0.5, alpha = 1, label='Mean' )

    axs[i].tick_params(labelsize=4, colors='k')    
    axs[i].legend(loc='upper left', fontsize=4)
axs[46].axis('off')
axs[47].axis('off')

if use_abbrevs == True:

    titles = mask_3D.abbrevs.values

else:

    titles = np.array( [str( mask_3D.region.values[i]+1 ) for i in range(len(mask_3D.region.values))] )
    
for axs, title in zip(axs.flatten(),titles): axs.set_title(title, fontsize=8)
fig.autofmt_xdate()
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close()

#----------------------------------------------------------------------------
# PLOT: regionally-averaged timeseries gridcell statistics per region
#----------------------------------------------------------------------------

for i in range(n_regions):

    figstr = variable + '-' + 'ipcc-ar6-land-region-timeseries' + '-' + 'stats' + '-' + temporalstr + '-' + 'region' + '-' + str(i+1).zfill(2) + '.png'
    if use_yearly == True:
        titlestr = variable + ': gridcell statistics aggregated yearly over IPCC AR6 land region ' + str(i+1)
    else:
        titlestr = variable + ': gridcell statistics aggregated monthly over IPCC AR6 land region ' + str(i+1)
    if (variable == 'BA_Total') | (variable == 'Cem_Total'):
        ylabelstr = 'Total'
    else:
        ylabelstr = 'TC30_F'
    
    fig, ax = plt.subplots(figsize=(13.33,7.5))

    if use_smooth == True:
                
        plt.fill_between( Z.time.values, pd.Series( Z_regional_weighted_025.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), pd.Series( Z_regional_weighted_975.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), alpha=0.25, color='lightblue', label='95% CI' )
        plt.fill_between( Z.time.values, pd.Series( Z_regional_weighted_100.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), pd.Series( Z_regional_weighted_900.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), alpha=0.1, color='blue', label='90% CI' ) 
        plt.fill_between( Z.time.values, pd.Series( Z_regional_weighted_250.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), pd.Series( Z_regional_weighted_750.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), alpha=0.1, color='purple', label='IQR' ) 
        plt.plot( Z.time.values, pd.Series( Z_regional_weighted_median.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), color='purple', lw=0.5, alpha = 1, label='Median' )
        plt.plot( Z.time.values, pd.Series( Z_regional_weighted_mean.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), color='red', lw=0.5, alpha = 1, label='Mean' )

    else:

        plt.fill_between( Z.time.values, Z_regional_weighted_025.isel(region=i).values, Z_regional_weighted_975.isel(region=i).values, alpha=0.25, color='lightblue', label='95% CI' ) 
        plt.fill_between( Z.time.values, Z_regional_weighted_100.isel(region=i).values, Z_regional_weighted_900.isel(region=i).values, alpha=0.1, color='blue', label='90% CI' ) 
        plt.fill_between( Z.time.values, Z_regional_weighted_250.isel(region=i).values, Z_regional_weighted_750.isel(region=i).values, alpha=0.1, color='purple', label='IQR' ) 
        plt.plot( Z.time.values, Z_regional_weighted_median.isel(region=i).values, color='purple', lw=0.5, alpha = 1, label='Median' )
        plt.plot( Z.time.values, Z_regional_weighted_mean.isel(region=i).values, color='red', lw=0.5, alpha = 1, label='Mean' )

    plt.legend(loc='upper left', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize, colors='k')    
    plt.ylabel( ylabelstr, fontsize=fontsize)

    if use_abbrevs == True:
    
        titlestr = titlestr + ' (' + mask_3D.abbrevs.values[i] + ')'
    
    else:
    
        titlestr = titlestr + ' (' + str( mask_3D.region.values[i]+1 ) + ')'
        
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
    plt.close()

#----------------------------------------------------------------------------
print('** END')
