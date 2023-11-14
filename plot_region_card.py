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

ax = plt.axes(rect, projection=ccrs.PlateCarree(), )
ax.set_extent((110,160, -45, -10))
ax.stock_img()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
blank_axes(ax)

#------------------------------------------------------------------------------
# PLOT: location map and highlight spatial data inset region
#------------------------------------------------------------------------------

left, bottom, width, height = 0.02, 0, 0.16, 0.2
rect = [left,bottom,width,height]

ax2 = plt.axes(rect, projection=ccrs.PlateCarree(), )
ax2.set_global()
ax2.add_feature(cfeature.LAND)
ax2.add_feature(cfeature.OCEAN)
lon0,lon1,lat0,lat1 = ax.get_extent()
box_x = [lon0, lon1, lon1, lon0, lon0]
box_y = [lat0, lat0, lat1, lat1, lat0]
plt.plot(box_x, box_y, color='red', lw=0.5, transform=ccrs.PlateCarree())
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
ax6.text(0.5, 0.0,'IPCC Region Card', ha='center', color='white', fontsize=fontsize)
blank_axes(ax6)

#------------------------------------------------------------------------------
# PLOT: N arrow
#------------------------------------------------------------------------------

'''
left, bottom, width, height = 0.7, 0.7, 0.1, 0.05
rect = [left,bottom,width,height]

ax4 = plt.axes(rect)
ax4.set_facecolor(bg_color2)
ax4.text(0.5, 0.0,u'\u25B2 \nN ', ha='center', color='white', fontsize=10, family='Arial', rotation = 0)
blank_axes(ax4)

#------------------------------------------------------------------------------
# PLOT: legend
#------------------------------------------------------------------------------
# legends can be quite long, so set near top of map (0.4 - bottom + 0.5 height = 0.9 - near top)

left = 0
bottom = 0.4
width = 0.16
height = 0.5
rect = [left,bottom,width,height]
rect = [left,bottom,width,height]
ax5 = plt.axes(rect)

# create an array of color patches and associated names for drawing in a legend
colors = sorted(cartopy.feature.COLORS.keys())
handles = []
names = []
# for each cartopy defined color, draw a patch, append handle to list, and append color name to names list
for c in colors:
    patch = mpatches.Patch(color=cfeature.COLORS[c], label=c)
    handles.append(patch)
    names.append(c)

# do some example lines with colors
river = mlines.Line2D([], [], color=cfeature.COLORS['water'], marker='',
                              markersize=15, label='river')
coast = mlines.Line2D([], [], color='black', marker='',
                              markersize=15, label='coast')
bdy  = mlines.Line2D([], [], color='grey', marker='',
                      markersize=15, label='state boundary')
handles.append(river)
handles.append(coast)
handles.append(bdy)
names.append('river')
names.append('coast')
names.append('state boundary')

# create legend
ax5.legend(handles, names)
ax5.set_title('Legend',loc='left')
blank_axes(ax5)
'''

#------------------------------------------------------------------------------
# SAVE: region card
#------------------------------------------------------------------------------

plt.savefig( 'region_card.png', dpi=dpi )    

#------------------------------------------------------------------------------
print('** END')

