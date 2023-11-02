#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_highest_variance_gridcell_timeseries.py
#------------------------------------------------------------------------------
# Version 0.2
# 30 October, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

# Dataframe libraries:

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# Colour libraries:

import cmocean

# Plotting libraries:

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib import colors as mcolors
# %matplotlib inline # for Jupyter Notebooks

# Mapping libraries:

import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
dpi = 300

use_sd = True
use_log = False

variable = 'BA_Total'
#variable = 'BA_Forest_NonForest'
#variable = 'Cem_Total'
#variable = 'Cem_Forest_NonForest'

nc_file = 'OUT/' + variable + '.nc'

resolution = '10m'          # ['110m','50m','10m']
use_gridlines = True        # [True,False]
use_cmocean = True          # [True,False] False --> 'bwr' 
use_cyan = False            # [True,False] False --> 'grey'
use_projection = 'robinson' # see projection list below

# SET: projection
    
if use_projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0)
if use_projection == 'europp': p = ccrs.EuroPP()
if use_projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0)
if use_projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0)
if use_projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0)
if use_projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0)
if use_projection == 'northpolarstereo': p = ccrs.NorthPolarStereo()
if use_projection == 'orthographic': p = ccrs.Orthographic(0,0)
if use_projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0)
if use_projection == 'robinson': p = ccrs.Robinson(central_longitude=0)
if use_projection == 'southpolarstereo': p = ccrs.SouthPolarStereo()    

# LOAD: Natural Earth features
    
borders = cf.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resolution, facecolor='none', alpha=1)
land = cf.NaturalEarthFeature('physical', 'land', scale=resolution, edgecolor='k', facecolor=cf.COLORS['land'])
ocean = cf.NaturalEarthFeature('physical', 'ocean', scale=resolution, edgecolor='none', facecolor=cf.COLORS['water'])
lakes = cf.NaturalEarthFeature('physical', 'lakes', scale=resolution, edgecolor='b', facecolor=cf.COLORS['water'])
rivers = cf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resolution, edgecolor='b', facecolor='none')

# SET: cmap

if use_cmocean == True:
    cmap_full = cmocean.cm.thermal
    cmap = cmocean.tools.crop_by_percent(cmap_full, 20, which='both') # clip 20% from ends
else:
    #cmap = 'RdBu_r'
    cmap = 'plasma'

#----------------------------------------------------------------------------
# LOAD: netCDF4
#----------------------------------------------------------------------------
# NB: MODIS --> HDF5 has latitudes running [-90,+90] top to bottom in the array so flip them!

#ds_in = xr.load_dataset( nc_file )
#ds = ds_in.copy()
#ds = ds.assign(variables={variable:ds[variable][::-1]})
#ds = ds.assign_coords({"lat": ds.lat * -1})

ds = xr.load_dataset( nc_file )

#----------------------------------------------------------------------------
# FIND: highest variance or highest total gridcell
#----------------------------------------------------------------------------

lats = ds.lat.values
lons = ds.lon.values

if use_sd == True:
    
    criterion = 'sd'

    # COMPUTE: unbiased esimtate of SD per gridcell (along time axis)
    
    sd_map = ds[variable].std(skipna=True, ddof=1, dim="time")

    # FIND: gridcell containing max SD

    sd_max = sd_map.where(sd_map==sd_map.max(), drop=True).squeeze()
    lat_max = sd_max.lat.values + 0
    lon_max = sd_max.lon.values + 0

else:

    criterion = 'sum'

    # COMPUTE: sum per gridcell (along time axis)
    
    sum_map = ds[variable].sum(skipna=True, dim="time")
    
    # FIND: gridcell containing max sum

    sum_max = sum_map.where(sum_map==sum_map.max(), drop=True).squeeze()
    lat_max = sum_max.lat.values + 0
    lon_max = sum_max.lon.values + 0
    
# EXTRACT: timeseries

ts = ds[variable].sel(lon=lon_max, lat=lat_max, method="nearest")

#lat_max *= -1

#----------------------------------------------------------------------------
# PLOT: gridcell timeseries
#----------------------------------------------------------------------------

figstr = variable + '-' + 'timeseries' + '-' + criterion + '-' + '(' + str(lat_max) + 'N' + ',' + str(lon_max) + 'E' + ')' + '.png'
if criterion == 'sd':   
    titlestr = variable + ': highest variance gridcell (' + str(lat_max) + r'$^{\circ}$N' + ',' + str(lon_max) + r'$^{\circ}$E' + ')'
else:
    titlestr = variable + ': highest sum gridcell (' + str(lat_max) + r'$^{\circ}$N' + ',' + str(lon_max) + r'$^{\circ}$E' + ')'

if (variable == 'BA_Forest_NonForest') | (variable == 'Cem_Forest_NonForest'):
    ystr = 'TC30_F'
else:
    ystr = 'Total'             
           
fig, ax = plt.subplots(figsize=(13.33,7.5))     
plt.plot( ds.time, ts, ls='-', lw=1, color='k')
plt.ylabel(ystr, fontsize=fontsize, color='k')
plt.tick_params(labelsize=fontsize, colors='k')    
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#----------------------------------------------------------------------------
# PLOT: timeseries mean map + gridcell location
#----------------------------------------------------------------------------

Z = ds[variable].mean(axis=0)
   		  
if use_log == True:
    
    Z = Z + 1.0
    Z = np.log10(Z)

# SET: zeros to NaN

Z = Z.where(Z > 0.0)
    
#X,Y = np.meshgrid(lons,lats[::-1])
X,Y = np.meshgrid(lons,lats)
N = len(lons)*len(lats)

# TIDY: format

#x = X.reshape(N)
#y = Y.reshape(N)
#z = Z.reshape(N)
#df = pd.DataFrame({'lon':x, 'lat':y, 'variable':z}, index=range(N))

vmin = np.nanmin(Z)
vmax = np.nanmax(Z)

figstr = variable + '-' + 'map' + '-' + criterion + '-' + '(' + str(lat_max) + 'N' + ',' + str(lon_max) + 'E' + ')' + '.png'
if criterion == 'sd':   
    titlestr = variable + ': highest variance gridcell (' + str(lat_max) + r'$^{\circ}$N' + ',' + str(lon_max) + r'$^{\circ}$E' + ')'
else:
    titlestr = variable + ': highest sum gridcell (' + str(lat_max) + r'$^{\circ}$N' + ',' + str(lon_max) + r'$^{\circ}$E' + ')'

if use_log == True:    
    cbarstr = r'log$_{10}$(' + variable + ')'
else:
    cbarstr = variable
        
fig, ax = plt.subplots(figsize=(13.33,7.5), subplot_kw=dict(projection=p))    
# PowerPoint:            fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=144); plt.savefig('figure.png', bbox_inches='tight')
# Posters  (vectorized): fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=600); plt.savefig('my_figure.svg', bbox_inches='tight')                          
# Journals (vectorized): fontsize = 18; fig = plt.figure(figsize=(3.54,3.54), dpi=300); plt.savefig('my_figure.svg', bbox_inches='tight')     

#ax.set_extent([np.round(X.min()-5,0),np.round(X.max()+5,0),np.round(Y.min()-5,0),np.round(Y.max()+5,0)], crs=ccrs.PlateCarree())
ax.set_global()  
#ax.stock_img()                                                                
ax.add_feature(land, facecolor='black', linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=1)
if use_cyan == True:
    ax.add_feature(ocean, facecolor='cyan', alpha=1, zorder=1)
else:
    ax.add_feature(ocean, facecolor='grey', alpha=0.7, zorder=1)
#ax.add_feature(lakes)
#ax.add_feature(rivers, linewidth=0.5)
#ax.add_feature(borders, linestyle='-', linewidth=2, edgecolor='k', alpha=1, zorder=2)         
                
ax.coastlines(resolution=resolution, color='k', linestyle='-', linewidth=0.2, edgecolor='k', alpha=1, zorder=2)                                                                                  
ax.add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='white', alpha=1, zorder=3)         
                  
if use_gridlines == True:
            
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color='purple', alpha=1, linestyle='-', zorder=4)
    gl.xlines = True; gl.ylines = True
    gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,int((360/30)+1))) # every 5 degrees
    gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,int((180/30)+1)))   # every 5 degrees
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
          
#plt.scatter(x=X, y=Y, c=Z.values, marker='s', s=0.25, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), zorder=10)                 
g = Z.plot(ax=ax, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation':'horizontal','extend':'max','shrink':0.5, 'pad':0.05}, zorder=10)                 
cbar = g.colorbar; cbar.ax.tick_params(labelsize=fontsize); cbar.set_label(label=cbarstr, size=fontsize); cbar.ax.set_title(None, fontsize=fontsize)
        
plt.scatter( x = lon_max, y = lat_max, s=500, marker='o', facecolors='none', edgecolors='r', lw=3, alpha = 1, transform=ccrs.PlateCarree(), zorder=100)
                          
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close()

#----------------------------------------------------------------------------
print('** END')
