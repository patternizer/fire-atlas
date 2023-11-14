#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_ipcc_ar6_region_classifications.py
#------------------------------------------------------------------------------
# Version 0.2
# 13 November, 2023
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

import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from shapely.geometry.polygon import LinearRing

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

extract_classifications = False
plot_classifications = False
plot_facetgrid = False
plot_region = False
plot_inset = True

fontsize = 16
dpi = 300
cmap = 'plasma'

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

def regions_to_csv( regiontypestr ):

    region_dict = eval( "regionmask.defined_regions." + regiontypestr )
    abbrevs = np.array(region_dict.abbrevs)
    names = np.array(region_dict.names)
    df = pd.DataFrame({'id':np.arange(len(region_dict))+1, 'abbrev':abbrevs, 'name':names})
    df.set_index('id', inplace=True)
    df.to_csv( regiontypestr + '.csv')

def plot_regions( regiontypestr, option_kws, p):
    
    figstr = regiontypestr + '.png'
    fig, ax = plt.subplots( figsize = (13.33,7.5), subplot_kw = dict( projection = p ) )     
    eval( "regionmask.defined_regions." + regiontypestr + ".plot(" + option_kws + ")" )
    ax.set_global()
    plt.title( regiontypestr.title(), fontsize=16)
    plt.savefig(figstr, dpi=300, bbox_inches='tight')
    plt.close('all')

if extract_classifications == True:
    
    #----------------------------------------------------------------------------
    # EXTRACT: region classifications
    #----------------------------------------------------------------------------
        
    # SAVE; region code dictionaries
    
    regions_to_csv( 'giorgi' )
    regions_to_csv( 'srex' )
    regions_to_csv( 'ar6.land' )
    regions_to_csv( 'ar6.ocean' )
    regions_to_csv( 'ar6.all' )
    regions_to_csv( 'natural_earth_v5_0_0.countries_110' )

if plot_classifications == True:
    
    #----------------------------------------------------------------------------
    # PLOT: region classifications
    #----------------------------------------------------------------------------
    
    text_kws = dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w"))
    
    plot_regions( 'giorgi', "text_kws=text_kws, add_ocean=True, label_multipolygon='all'", p)
    plot_regions( 'srex', "text_kws=text_kws, add_ocean=True, label_multipolygon='all'", p)
    #plot_regions( 'ar6.land', "text_kws=text_kws, add_ocean=True, label_multipolygon='all'", p)
    #plot_regions( 'ar6.land', "text_kws=text_kws, add_ocean=True, add_land=True, add_label=False", p)
    plot_regions( 'ar6.land', "text_kws=text_kws, add_ocean=True, add_land=True, label='abbrev'", p)
    plot_regions( 'ar6.ocean', "text_kws=text_kws, add_land=True, label_multipolygon='all'", p)
    #plot_regions( 'ar6.all', "text_kws=text_kws, label_multipolygon='all'", p)
    plot_regions( 'ar6.all', "text_kws=text_kws, add_land=True, add_ocean=True, add_label=False", p)
    plot_regions( 'natural_earth_v5_0_0.countries_110', "text_kws=text_kws, add_ocean=True, add_label=True", p)

if plot_facetgrid == True:
    
    #----------------------------------------------------------------------------
    # EXTRACT: IPCC AR6 region masks and plot as facetgrid
    #----------------------------------------------------------------------------
    
    xr.set_options(display_style="text", display_expand_data=False, display_width=60)
    #cmap_mask = mcolors.ListedColormap(["none", "#9ecae1"])
    region_mask = regionmask.defined_regions.ar6.land.mask_3D(lons, lats)
    
    # PLOT: facetgrid of region masks
    
    figstr = 'ipcc-ar6-land-region-masks.png'
    titlestr = 'IPCC AR6 land region masks (1-46)'
    
    fig, ax = plt.subplots(figsize=(13.33,7.5), subplot_kw=dict(projection=p))    
    fg = region_mask.isel(region=slice(48)).plot(
        subplot_kws=dict(projection=ccrs.PlateCarree()),
        col="region",
        col_wrap=8,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        aspect=1.5,
        #cmap=cmap_mask,
        cmap=cmap,
    )
    for ax in fg.axes.flatten(): ax.coastlines()
    titles = np.array( [str( region_mask.region.values[i]+1 ) for i in range(len(region_mask.region.values))] )
    for ax, title in zip(fg.axes.flatten(),titles): ax.set_title(title, fontsize=fontsize)
    fg.fig.subplots_adjust(hspace=0, wspace=0.05);
    plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
    plt.close()

if plot_region == True:
    
    #----------------------------------------------------------------------------
    # PLOT: each local region
    #----------------------------------------------------------------------------

    text_kws = dict(color="#67000d", fontsize=16, bbox=dict(pad=0.2, color="w"))

    ar6_land = regionmask.defined_regions.ar6.land
    region_mask = regionmask.defined_regions.ar6.land.mask_3D(lons, lats)    
    region_names = regionmask.defined_regions.ar6.land.names
    region_abbrevs = regionmask.defined_regions.ar6.land.abbrevs
        
    for i in region_mask.region.values:
        
        figstr = 'ipcc-ar6-land-region-' + str(i+1).zfill(2) + '.png'
        titlestr = region_names[i] 
        
        fig, ax = plt.subplots(figsize=(13.33,7.5), subplot_kw=dict(projection=p))
        ax = ar6_land[[i]].plot(
            add_ocean=True,
            add_land=True,
            resolution="50m",
            projection=p,
            label="abbrev",
            text_kws=text_kws,
        )
        ax.set_title(titlestr, fontsize=fontsize)
        plt.savefig(figstr, dpi=dpi)
        plt.close()

if plot_inset == True:

    #----------------------------------------------------------------------------
    # PLOT: each local region as global map inset
    #----------------------------------------------------------------------------

    text_kws = dict(color="#67000d", fontsize=16, bbox=dict(pad=0.2, color="w"))
    
    ar6_land = regionmask.defined_regions.ar6.land
    region_mask = regionmask.defined_regions.ar6.land.mask_3D(lons, lats)        
    region_names = regionmask.defined_regions.ar6.land.names
    region_abbrevs = regionmask.defined_regions.ar6.land.abbrevs
    
    for i in region_mask.region.values:
                
        lats = region_mask.isel(region=i).lat.values
        lons = region_mask.isel(region=i).lon.values
        Y,X = np.meshgrid( lons, lats )    
        mask = region_mask.isel(region=i)
        latmin = X[mask].min()
        latmax = X[mask].max()
        lonmin = Y[mask].min()
        lonmax = Y[mask].max()

        figstr = 'ipcc-ar6-land-region-' + str(i+1).zfill(2) + '.png'
        titlestr = region_names[i] 
        
        fig = plt.figure(figsize=(13.33,7.5))        
        ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(), frameon=False)   # main container
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())                   # map region container
        ar6_land[[i]].plot(
            add_ocean=True,
            add_land=True,
            resolution="50m",
            projection=p,
            label="abbrev",
            text_kws=text_kws,
        )
        
        # SET: inset location relative to main plot (ax) in normalized units
        
        inset_x = 1
        inset_y = 1
        inset_size = 0.2
        
        ax2 = plt.axes([0, 0, 1, 1], projection=p)
        ax2.set_global()
        ax2.add_feature(cfeature.LAND)
        ax2.add_feature(cfeature.OCEAN)
        ax2.add_feature(cfeature.COASTLINE)

        ip = InsetPosition(ax1, [inset_x - inset_size / 2,
                                inset_y - inset_size / 2,
                                inset_size,
                                inset_size])
        ax2.set_axes_locator(ip)

              
        # ADD: selected region highlight to world map
        
        nvert = 100
        lons = np.r_[np.linspace(lonmin, lonmin, nvert),
                     np.linspace(lonmin, lonmax, nvert),
                     np.linspace(lonmax, lonmax, nvert)].tolist()
        lats = np.r_[np.linspace(latmin, latmax, nvert),
                     np.linspace(latmax, latmax, nvert),
                     np.linspace(latmax, latmin, nvert)].tolist()
        
        ring = LinearRing(list(zip(lons, lats)))
        ax2.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=1)
        
        ax1.set_title(titlestr, fontsize=fontsize)
        plt.savefig(figstr, dpi=dpi)
        plt.close()
    

#----------------------------------------------------------------------------
print('** END')

