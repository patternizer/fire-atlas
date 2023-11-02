#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: create_ipcc_ar6_region_svg.py
#------------------------------------------------------------------------------
# Version 0.1
# 2 November, 2023
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

# Mapping libraries:

import cartopy.crs as ccrs

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

#regiontype = 'giorgi'
#regiontype = 'srex'
regiontype = 'ar6.land'
#regiontype = 'ar6.ocean'
#regiontype = 'ar6.all'
#regiontype = 'natural_earth_v5_0_0.countries_110'

dpi = 300
filetype = 'svg'

latstep = 0.25
lonstep = 0.25
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lats = lats[::-1]

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

#----------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------

def plot_regions( regiontypestr, option_kws, p, filetype, dpi):
    
    figstr = regiontypestr + '.' + filetype
    fig, ax = plt.subplots( figsize = (13.33,7.5), subplot_kw = dict( projection = p ) )     
    eval( "regionmask.defined_regions." + regiontypestr + ".plot(" + option_kws + ")" )
    ax.set_global()
    plt.savefig(figstr, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close('all')

#----------------------------------------------------------------------------
# PLOT: region classifications
#----------------------------------------------------------------------------

plot_regions( 'ar6.land', "add_land=True, add_ocean=True, add_label=False", p, filetype, dpi)

#----------------------------------------------------------------------------
print('** END')

