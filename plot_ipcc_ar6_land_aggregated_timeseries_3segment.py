#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_ipcc_ar6_land_aggregated_timeseries.py
#------------------------------------------------------------------------------
# Version 0.4
# 4 November, 2023
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

# Stats libraries:

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
import scipy
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from lineartree import LinearTreeClassifier, LinearTreeRegressor

import pwlf
import random
import itertools

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
dpi = 300

use_abbrevs = True

plot_facetgrid = False

variable = 'BA_Total'
#variable = 'BA_Forest_NonForest'
#variable = 'Cem_Total'
#variable = 'Cem_Forest_NonForest'

nc_file = 'OUT/' + variable + '.nc'

use_yearly = False
if use_yearly == True:
    temporalstr = 'yearly'
    nsmooth = 5
    use_all_but_last_year = False
    method = 'theil_sen'    
else:
    temporalstr = 'monthly'
    nsmooth = 60
    use_all_but_last_year = False
    method = 'ols' # Theil-Sen fails to converge for "flat" LOESS in monthly series

# LTR parameters:

depth = 5

# OLS regression parameters:
    
#method = 'ols'    
#method = 'theil_sen'    
#method = 'ransac'    

nbreaks_max = 2

#----------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------

def linear_regression_ols( x, y, method ):
    
    if method == 'ols':
        regr = linear_model.LinearRegression()
    elif method == 'theil_sen':
        regr = TheilSenRegressor(random_state=42)
    elif method == 'ransac':
        regr = RANSACRegressor(random_state=42)
    
    n = len(x)
    x = np.arange(n)
    X = x.reshape(n,1)
    
    regr.fit(X, y)
    ypred = regr.predict(X.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_
    
    return t, ypred, slope, intercept

def adjusted_r_squared(x,y):
    
    X = x.reshape(len(x),1)
    model = sm.OLS(y, X).fit()
    R2adj = model.rsquared_adj

    return R2adj

def calculate_piecewise_regression( x, y, depth, nsmooth ):
    
    # FIT: linear tree regressor
                    
    mask = np.isfinite(y)
    n_obs = mask.sum()
    x_obs = np.arange( n_obs ) / n_obs
    x_obs = x_obs[mask].reshape(-1, 1)
    y_obs = y[mask].reshape(-1, 1)		     
    
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        max_depth = depth,
        min_samples_split = np.max([6, nsmooth]),  # >5 or float[0,1]
        min_samples_leaf = np.max([3, nsmooth]),   # >2 or float[0,1]
        max_bins = 10,
        min_impurity_decrease = 0.0
    ).fit(x_obs, y_obs)    
    y_fit = lt.predict(x_obs)           
        
    # FIT: decision tree regressor

    dr = DecisionTreeRegressor(   
       max_depth = depth
    ).fit(x_obs, y_obs)
    x_fit = dr.predict(x_obs)

    # COMPUTE: goodness of fit

    mask_ols = np.isfinite(y_obs) & np.isfinite(y_fit.reshape(-1,1))
    corrcoef = scipy.stats.pearsonr(y_obs[mask_ols], y_fit.reshape(-1,1)[mask_ols])[0]
    R2adj = adjusted_r_squared(y_obs, y_fit.reshape(-1,1))
    
    return x_fit, y_fit, corrcoef, R2adj

#----------------------------------------------------------------------------
# LOAD: netCDF4
#----------------------------------------------------------------------------

ds = xr.load_dataset( nc_file )

#----------------------------------------------------------------------------
# EXTRACT: regional mean timeseries and compute statistics
#----------------------------------------------------------------------------

Z = ds[variable]

if use_yearly == True:

    Z = Z.resample(time="AS").sum(dim="time")   

    if use_all_but_last_year == True:    
        
        Z = Z.sel(time=slice(Z.time[0],Z.time[-2]))

# SET: zeros to NaN

#Z = Z.where(Z > 0.0)

mask_3D = regionmask.defined_regions.ar6.land.mask_3D( Z )
n_regions = mask_3D.region.shape[0]

#weights = np.cos(np.deg2rad(Z.lat))
#Z_regional_weighted_sum = Z.weighted(mask_3D * weights).sum(dim=("lat", "lon"))
Z_regional_weighted_sum = Z.weighted(mask_3D).sum(dim=("lat", "lon"))

#----------------------------------------------------------------------------
# PLOT: facetgrid of regionally-averaged timeseries gridcell stats
#----------------------------------------------------------------------------

if plot_facetgrid == True:
    
    figstr = variable + '-' + 'ipcc-ar6-land-region-timeseries' + '-' + 'sum' + '-' + temporalstr + '.png'
    titlestr = variable + ': aggregated sum across each IPCC AR6 land region (1-46)'
    
    fig, axs = plt.subplots(ncols=8, nrows=6, figsize=(13.33,7.5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.3, 'wspace': 0.1})
    axs = axs.ravel()
    for i in range(n_regions):        
        axs[i].plot( Z.time.values, Z_regional_weighted_sum.isel(region=i).values, color='black', lw=1, alpha = 1 )    
        axs[i].tick_params(labelsize=4, colors='k')    
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
    
    print('Region=', i)
    
    # SMOOTH: with a Gaussian filter (windowed)
    
    t = Z.time.values
    ts = Z_regional_weighted_sum.isel(region=i).values    
    ts_before_mean = np.nanmean( ts[0:int(nsmooth/2)] )
    ts_after_mean = np.nanmean( ts[-int(nsmooth/2)-1:] )
    ts_windowed = np.hstack([ np.ones( int(nsmooth/2) ) * ts_before_mean, ts, np.ones( int(nsmooth/2) ) * ts_after_mean ])
    ts_smoothed = pd.Series( ts_windowed ).rolling( nsmooth, center=True, win_type='gaussian').mean(std=3).values[int(nsmooth/2):-int(nsmooth/2):]

    # LOESS fit to smooth
        
    y_loess = sm.nonparametric.lowess( exog=t, endog=ts_smoothed, frac=0.6 )[:,1]

    # STL fit --> trend
        
    #trend, seasonal, residual = ts_decomposition( ts, freq )

    # OLS fit to raw data

    t_ols, y_ols, slope_ols, intercept_ols = linear_regression_ols( t, ts, method )

    # LTR fit to LOESS

    x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( t, y_loess, depth, nsmooth )

    # EXTRACT: segment breaks

    df_delta_abs = pd.DataFrame( {'delta_abs': np.hstack([ 0, np.round( np.abs( np.diff( y_fit, 2 )), 0 )  ]).ravel()} )
    idx_breaks = df_delta_abs[df_delta_abs['delta_abs'] > 0].index.to_list()   
              
    if len(idx_breaks) > 2:
    
        # CONSTRAIN: to nbreaks_max --> (nbreaks_max+1) segments in the construction of the LTR correlation to the LOESS fit
        
        combinations = list(itertools.combinations(idx_breaks, nbreaks_max))

        # LOOP: over combinations of unique sets of breakpoints    

        correlations = []    
        for combo in combinations:

            # SORT: breakpoints in combination            

            random_breakpoints = np.sort( combo ) 

            # FIT: to LOESS
    
            random_ts = []
            for k in range(len(random_breakpoints)):
                if k == 0:
                    t_segment = t[0:random_breakpoints[k]]
                    ts_segment = y_fit[0:random_breakpoints[k]]
                    t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
                    random_ts.append(ts_segment_ols)
                else: 
                    t_segment = t[random_breakpoints[k-1]:random_breakpoints[k]]
                    ts_segment = y_fit[random_breakpoints[k-1]:random_breakpoints[k]]                
                    t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
                    random_ts.append(ts_segment_ols)    
            t_segment = t[random_breakpoints[k]:]
            ts_segment = y_loess[random_breakpoints[k]:]
            t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
            random_ts.append(ts_segment_ols)

            # JOIN: segments
            
            random_ts = np.hstack( random_ts )
            
            # COMPUTE: correlation of segmented fit on LOESS
            
            correlation = np.corrcoef( random_ts, y_loess)[0, 1]
            correlations.append( correlation )
        
        # EXTRACT: best combination
        
        best_combo = combinations[ np.argmax( correlations ) ]        
        idx_breaks = list( best_combo )

        # FIT: optimum curve
            
        y_best = []
        for k in range(len(idx_breaks)):            
            if k == 0:
                t_segment = t[0:idx_breaks[k]]
                ts_segment = y_loess[0:idx_breaks[k]]
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)
            else: 
                t_segment = t[idx_breaks[k-1]:idx_breaks[k]]
                ts_segment = y_loess[idx_breaks[k-1]:idx_breaks[k]]                
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)    
        t_segment = t[idx_breaks[k]:]
        ts_segment = y_loess[idx_breaks[k]:]
        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
        y_best.append(ts_segment_ols)

        # JOIN: segments
                
        y_best = np.hstack( y_best )        
    
    elif len(idx_breaks) == 2:

        # FIT: optimum curve
        
        y_best = []
        for k in range(len(idx_breaks)):            
            if k == 0:
                t_segment = t[0:idx_breaks[k]]
                ts_segment = y_loess[0:idx_breaks[k]]
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)
            else: 
                t_segment = t[idx_breaks[k-1]:idx_breaks[k]]
                ts_segment = y_loess[idx_breaks[k-1]:idx_breaks[k]]                
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)    
        t_segment = t[idx_breaks[k]:]
        ts_segment = y_loess[idx_breaks[k]:]
        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
        y_best.append(ts_segment_ols)

        # JOIN: segments
                
        y_best = np.hstack( y_best )        
        
    elif len(idx_breaks) == 1:
        
        # FIT: optimum curve
        
        y_best = []
        t_segment = t[0:idx_breaks[k]]
        ts_segment = y_loess[0:idx_breaks[k]]
        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
        y_best.append(ts_segment_ols)
        t_segment = t[idx_breaks[k]:]
        ts_segment = y_loess[idx_breaks[k]:]
        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
        y_best.append(ts_segment_ols)

        # JOIN: segments
                
        y_best = np.hstack( y_best )        

    elif len(idx_breaks) == 0:
        
        # FIT: optimum curve
        
        t_segment = t
        ts_segment = y_loess
        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_ols = linear_regression_ols( t_segment, ts_segment, method )
        y_best = ts_segment_ols
            
    figstr = variable + '-' + 'ipcc-ar6-land-region-timeseries' + '-' + 'sum' + '-' + temporalstr + '-' + 'region' + '-' + str(i+1).zfill(2) + '.png'
    if use_yearly == True:
        titlestr = variable + ': aggregated yearly total over IPCC AR6 land region ' + str(i+1)
    else:
        titlestr = variable + ': aggregated monthly total over IPCC AR6 land region ' + str(i+1)
    if (variable == 'BA_Total') | (variable == 'Cem_Total'):
        ylabelstr = 'Total'
    else:
        ylabelstr = 'TC30_F'
        
    # PLOT: timeseries and fits
    
    fig, ax = plt.subplots(figsize=(13.33,7.5))
    plt.plot( t, Z_regional_weighted_sum.isel(region=i).values, color='black', lw=3, alpha = 1, label='Sum' )        
    if use_yearly == True:
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='Theil-Sen' )
    else:
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='OLS' )
    plt.plot( t, y_loess, color='blue', lw=3, alpha = 1, label='LOESS' )        
    #plt.plot( t, y_fit, color='yellow', lw=2, alpha = 1, label='LTR fit (n=' + str(len(idx_all)+1) + ')' )        

    if len(idx_breaks) > 1:            
        for k in range(len(idx_breaks)):                
            plt.axvline( x = t[ idx_breaks[k] ], ls='--', color='black')                
            if  k == 0:
                plt.plot( t[0:idx_breaks[k]], y_best[0:idx_breaks[k]], color='yellow', lw=3, alpha = 1, label='Best fit LTR segmentation (n=' + str(len(idx_breaks)+1) + ')' )
            else:
                plt.plot( t[idx_breaks[k-1]:idx_breaks[k]], y_best[idx_breaks[k-1]:idx_breaks[k]], color='yellow', lw=3, alpha = 1)                
        plt.plot( t[idx_breaks[k]:], y_best[idx_breaks[k]:], color='yellow', lw=3, alpha = 1)                                               
    elif len(idx_breaks) == 1:
        for k in range(len(idx_breaks)):                
            plt.axvline( x = t[ idx_breaks[k] ], ls='--', color='black')                
            plt.plot( t[0:idx_breaks[k]], y_best[0:idx_breaks[k]], color='yellow', lw=3, alpha = 1, label='Best fit LTR segmentation (n=' + str(len(idx_breaks)+1) + ')' )
        plt.plot( t[idx_breaks[k]:], y_best[idx_breaks[k]:], color='yellow', lw=3, alpha = 1)                                               
    elif len(idx_breaks) == 0:
        plt.plot( t, y_best, color='yellow', lw=3, alpha = 1, label='Best fit LTR segmentation (n=' + str(len(idx_breaks)+1) + ')' )

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
