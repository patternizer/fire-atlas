#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_region_card.py
#------------------------------------------------------------------------------
# Version 0.2
# 14 November, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

# Dataframe libraries:

import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import datetime

# Plotting libraries:

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Mapping libraries:

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

fontsize = 10
dpi = 300
bg_color = 'red'
bg_color2 = 'yellow'

latstep = 0.25
lonstep = 0.25
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lats = lats[::-1]

use_projection = 'platecarree' # see projection list below

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

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def blank_axes(ax):
    """
    blank axis spines, tick marks and tick labels for ax
    """

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off' ,\
                        bottom='off', top='off', left='off', right='off' )
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

#------------------------------------------------------------------------------
# LOAD: IPCC AR6 land
#------------------------------------------------------------------------------

text_kws = dict(color="#67000d", fontsize=16, bbox=dict(pad=0.2, color="w"))    

ar6_land = regionmask.defined_regions.ar6.land
region_mask = regionmask.defined_regions.ar6.land.mask_3D(lons, lats)        
region_names = regionmask.defined_regions.ar6.land.names
region_abbrevs = regionmask.defined_regions.ar6.land.abbrevs

#------------------------------------------------------------------------------
# LOOP: over regions
#------------------------------------------------------------------------------

for i in region_mask.region.values:
                
    lats = region_mask.isel(region=i).lat.values
    lons = region_mask.isel(region=i).lon.values
    Y,X = np.meshgrid( lons, lats )    
    mask = region_mask.isel(region=i)
    latmin = X[mask].min()
    latmax = X[mask].max()
    lonmin = Y[mask].min()
    lonmax = Y[mask].max()

    figstr = 'ipcc-ar6-land-region-card-' + str(i+1).zfill(2) + '.png'
    titlestr = region_names[i] 

    #------------------------------------------------------------------------------
    # INITIALISE: figure instance (outer frame)
    #------------------------------------------------------------------------------
    
    cm = 1 / 2.54
    fig = plt.figure(figsize=(5*cm,9*cm))        
            
    left, bottom, width, height = -0.05, -0.05, 1.1, 1.05
    rect = [left,bottom,width,height]
    ax3 = plt.axes(rect)
    blank_axes(ax3)
    
    # SET: desired axis elements (optional)
    
    ax3.spines['right'].set_visible(True)
    ax3.spines['top'].set_visible(True)
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['left'].set_visible(True)
    ax3.text(0.15, 0.07, 'Michael Taylor, CRU/UEA -- ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fontsize=4)
    
    # DRAW: rounded rectangle inside filled background rectangle
    
    rounded_rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle='Round, pad=0, rounding_size=0.05', color=bg_color2, alpha = 0.5)
    rect = Rectangle((0, 0), width, 1, color=bg_color)
    ax3.add_patch(rect)
    ax3.add_patch(rounded_rect)
        
    #------------------------------------------------------------------------------
    # PLOT: spatial data
    #------------------------------------------------------------------------------
    
    left, bottom, width, height = 0.2, 0, 0.7, 0.9
    rect = [left,bottom,width,height]
    
    ax = plt.axes(rect, projection = p)
    ar6_land[[i]].plot(
            add_ocean=True,
            add_land=True,
            resolution="50m",
            projection=p,
            label="abbrev",
            text_kws=text_kws,
    )
    #ax.set_extent((110,160, -45, -10))
    #ax.stock_img()
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    #ax.add_feature(cfeature.COASTLINE)
    blank_axes(ax)
    
    #------------------------------------------------------------------------------
    # PLOT: location map and highlight spatial data inset region
    #------------------------------------------------------------------------------
    
    left, bottom, width, height = 0.02, 0, 0.16, 0.2
    rect = [left,bottom,width,height]
    
    ax2 = plt.axes(rect, projection = p)
    ax2.set_global()
    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.OCEAN)
    lon0,lon1,lat0,lat1 = ax.get_extent()
    box_x = [lon0, lon1, lon1, lon0, lon0]
    box_y = [lat0, lat0, lat1, lat1, lat0]
    plt.plot(box_x, box_y, color='red', lw=0.5, transform=p)
    blank_axes(ax2)
    
    #------------------------------------------------------------------------------
    # PLOT: inset connector lines
    #------------------------------------------------------------------------------
    
    mark_inset(ax2, ax, loc1=2, loc2=4, fc="none", ec="0.5")
    
    #------------------------------------------------------------------------------
    # PLOT: title
    #------------------------------------------------------------------------------
    
    left, bottom, width, height = 0.1, 0.95, 0.8, 0.04
    rect = [left,bottom,width,height]
    
    ax6 = plt.axes(rect)
    ax6.set_facecolor(bg_color)
    ax6.text(0.5, 0.0, titlestr, ha='center', color='white', fontsize=fontsize)
    blank_axes(ax6)
    
    #------------------------------------------------------------------------------
    # SAVE: region card
    #------------------------------------------------------------------------------
    
    plt.savefig( figstr, dpi=dpi )    

#------------------------------------------------------------------------------
print('** END')


