#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_hdf5.py
#------------------------------------------------------------------------------
# Version 0.1
# 24 October, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import netCDF4
import h5py

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

import matplotlib.gridspec as gridspec
import matplotlib.image as image

# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

file_hdf5 = 'DATA/BA_Qdeg_C6_202306.hdf5'
variable = 'BA_pixels'

fontsize = 10
#vmin = -6.0        
#vmax = 6.0    

#year_start = 1781
#year_end = 2023

latstep = 0.25
lonstep = 0.25
n_lat = int(180/latstep)
n_lon = int(360/lonstep)        

lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )

dpi = 300                   # [144,300,600]
resolution = '10m'          # ['110m','50m','10m']
use_gridlines = True        # [True,False]
use_cmocean = True          # [True,False] False --> 'bwr' 
use_cyan = False             # [True,False] False --> 'grey'
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
        
# CALCULATE: current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

#----------------------------------------------------------------------------
# METHODS:
#----------------------------------------------------------------------------
    
def array_to_xarray(lat, lon, da):

    xda = xr.DataArray(
        da,
        dims=[ "latitude", "longitude" ],
        coords={ "latitude": lat, "longitude": lon },
        attrs={ "long_name": "data_per_pixel", "description": "data per pixel", "units": "",},
    )
    return xda
        
#----------------------------------------------------------------------------
# LOAD: BA_Qdeg_C6_202306.hdf5 (0.25 x 0.25 degrees)
#----------------------------------------------------------------------------

print('loading data ...')

f = h5py.File( file_hdf5, 'r+' )
da = np.array( f[ variable ] )
da[da == 0.0] = np.nan      

lats = lats[::-1]

da_xr = array_to_xarray( lats, lons, da)

X,Y = np.meshgrid(lons,lats)
Z = da
N = len(lons)*len(lats)

# create 1D-arrays from 2D-arrays
x = X.reshape(N)
y = Y.reshape(N)
z = Z.reshape(N)

#df = pd.DataFrame({'lon':x, 'lat':y, 'variable':z}, index=range(N))

vmin = np.nanmin(da)
vmax = np.nanmax(da)
vmax = 1600.0

#==============================================================================
# PLOT: station map
#==============================================================================
            	
varstr = file_hdf5.split('/')[1].split('.')[0]
	    		            
figstr = varstr + '.png'        

# CREDITS
        
datastr = varstr + ('via Matt Jones and FireCrew')        
authorstr = r'$\bf{Graphic}$' + ': Michael Taylor, CRU/UEA' + ' -- ' + titletime
        
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
          
#plt.scatter(x=x, y=y, c=z, s=0.25, marker='s', alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), zorder=100)         
 
g = da_xr.plot(ax=ax, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation':'horizontal','extend':'max','shrink':0.5, 'pad':0.05}, zorder=10)                 
cb = g.colorbar; cb.ax.tick_params(labelsize=fontsize); cb.set_label(label=varstr, size=fontsize); cb.ax.set_title(None, fontsize=fontsize)
   
#temp_cyc, lon_cyc = add_cyclic_point(temp, coord=lon)

#if dpi == 144: xstart = 180; ystart=10; ystep = 20
#elif dpi == 300: xstart = 180; ystart=10; ystep = 40
#elif dpi == 600: xstart = 180; ystart=10; ystep = 80       
#plt.annotate(datastr, xy=(xstart,ystart+ystep*2), xycoords='figure pixels', color='k', fontsize=8) 
#plt.annotate(authorstr, xy=(xstart,ystart+ystep*1), xycoords='figure pixels', color='k', fontsize=8)   
                       
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)
print("matplotlib : ", matplotlib.__version__)
print("cartopy    : ", cartopy.__version__)

# -----------------------------------------------------------------------------
print('** END')
