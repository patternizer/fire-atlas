#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_ipcc_ar6_land_aggregated_timeseries.py
#------------------------------------------------------------------------------
# Version 0.5
# 20 November, 2023
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

#from statsmodels.tsa.seasonal import STL
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
# METHODS
#----------------------------------------------------------------------------

'''
def ts_decomposition( ts, freq ):
       
    #model = STL( ts, period=None, seasonal=7, trend=None, low_pass=None, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False, seasonal_jump=1, trend_jump=1, low_pass_jump=1).fit()
    
    # seasonal: must be odd and >=3
    
    model = STL( ts, period=freq, seasonal=freq+1, low_pass=freq+1, robust=True).fit()
    trend = model.trend
    seasonal = model.seasonal
    residual = model.resid

    return trend, seasonal, residual
'''

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

    mask = np.isfinite(y_obs) & np.isfinite(y_fit.reshape(-1,1))
    corrcoef_fit = scipy.stats.pearsonr(y_obs[mask], y_fit.reshape(-1,1)[mask])[0]
    R2adj_fit = adjusted_r_squared(y_obs, y_fit.reshape(-1,1))
    
    return x_fit, y_fit, corrcoef_fit, R2adj_fit

def calculate_slope(x1, y1, x2, y2):
    
    #slope = ( y2[-1] - y1[-1] ) / ( x2[-1] - x1[-1] )
    slope = ( y2 - y1 ) / ( x2 - x1 )
    return slope

def prune_breakpoints(time_series, breakpoints, slope_threshold):

    pruned_breakpoints = []
    slopes_pruned_breakpoints = []
    slopes_all = []
    for k in range(len(breakpoints)):

        # NORMALISE: to max value        

        time_series_max = np.nanmax( time_series )
        time_series = time_series / time_series_max
        
        if k == 0:

            x1, y1 = 0, time_series[0]
            x2, y2 = breakpoints[k], time_series[breakpoints[k]]
            x3, y3 = breakpoints[k + 1], time_series[breakpoints[k + 1]]
            
            slope1 = calculate_slope(x1, y1, x2, y2)
            slope2 = calculate_slope(x2, y2, x3, y3)
            #slope_change = abs(slope2 - slope1)
            slope_change = abs( abs(slope2) - abs(slope1) )
            slopes_all.append(slope_change)
            print(k,slope_change)
            if slope_change >= slope_threshold:
                pruned_breakpoints.append(x2)
                slopes_pruned_breakpoints.append(slope_change)

        elif k == len(breakpoints)-1:
            
            x1, y1 = breakpoints[k - 1], time_series[breakpoints[k - 1]]
            x2, y2 = breakpoints[k], time_series[breakpoints[k]]
            x3, y3 = len(time_series), time_series[-1]
            
            slope1 = calculate_slope(x1, y1, x2, y2)
            slope2 = calculate_slope(x2, y2, x3, y3)
            slope_change = abs( abs(slope2) - abs(slope1) )
            slopes_all.append(slope_change)
            print(k,slope_change)
            if slope_change >= slope_threshold:
                pruned_breakpoints.append(x2)
                slopes_pruned_breakpoints.append(slope_change)

        elif (k>0) & (k<len(breakpoints)-1):

            x1, y1 = breakpoints[k - 1], time_series[breakpoints[k - 1]]
            x2, y2 = breakpoints[k], time_series[breakpoints[k]]
            x3, y3 = breakpoints[k + 1], time_series[breakpoints[k + 1]]
            slope1 = calculate_slope(x1, y1, x2, y2)
            slope2 = calculate_slope(x2, y2, x3, y3)
            slope_change = abs( abs(slope2) - abs(slope1) )
            slopes_all.append(slope_change)
            print(k,slope_change)
            if slope_change >= slope_threshold:
                pruned_breakpoints.append(x2)
                slopes_pruned_breakpoints.append(slope_change)

    return pruned_breakpoints, slopes_pruned_breakpoints, slopes_all

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
dpi = 300

use_log = False
use_abbrevs = True
use_latitudinal_weighting = False
plot_facetgrid = True

# LOESS parameters:

loess_frac = 0.6

# LTR parameters:

depth = 5
nbreaks_max = 2
slope_threshold = 0.05

# OLS regression parameters:
    
#method = 'ols'    
#method = 'theil_sen'    
#method = 'ransac'    

variable_list = [ 'BA_Total', 'BA_Forest_NonForest', 'Cem_Total', 'Cem_Forest_NonForest' ]
timescale_list = [ 'yearly', 'yearly_jj', 'seasonal_mam', 'seasonal_jja', 'seasonal_son', 'seasonal_djf', 'monthly' ]

variable = variable_list[0]
timescale = timescale_list[0]

#----------------------------------------------------------------------------
# RUN:
#----------------------------------------------------------------------------

#for variable in variable_list:   
#    for timescale in timescale_list:

nc_file = 'OUT/' + variable + '.nc'

if timescale == 'yearly':

	nsmooth = 5
	use_all_but_last_year = False
	method = 'theil_sen'    
	
elif timescale == 'yearly_jj':

	nsmooth = 5
	use_all_but_last_year = False
	method = 'theil_sen'    

elif timescale == 'seasonal_mam':

	nsmooth = 5
	use_all_but_last_year = False
	method = 'theil_sen'    

elif timescale == 'seasonal_jja':

	nsmooth = 5
	use_all_but_last_year = False
	method = 'theil_sen'    

elif timescale == 'seasonal_son':

	nsmooth = 5
	use_all_but_last_year = False
	method = 'theil_sen'    

elif timescale == 'seasonal_djf':

	nsmooth = 5
	use_all_but_last_year = False
	method = 'theil_sen'    

elif timescale == 'monthly': 

	nsmooth = 60
	use_all_but_last_year = False
	method = 'ols' # Theil-Sen fails to converge for "flat" LOESS in monthly series

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

if timescale == 'yearly':

	Z = Z.resample(time="AS").sum(dim="time")   

	if use_all_but_last_year == True:    
		
		Z = Z.sel(time=slice(Z.time[0],Z.time[-2]))

elif timescale == 'yearly_jj':
	
	Z = Z.resample(time='AS-JUN').sum('time') # anchored offset for the austral year

	'''
	PANDAS: implementation:
		
	# DT.groupby(pd.DatetimeIndex(DT.Date).shift(-3,freq='m').year)    
	# DT.groupby(DT.index.shift(-3,freq='m').year)
	'''

elif timescale == 'seasonal_mam':

	month_length = Z.time.dt.days_in_month    
	weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
	Z_weighted = (Z * weights)
	Z = Z_weighted.sel(time=Z_weighted.time.dt.season == 'MAM')
	Z = Z.groupby(Z.time.dt.year).sum("time")
	Z = Z.rename({'year':'time'})            
	Z['time'] = pd.date_range(start=str(Z.time[0].values), end=str(Z.time[-1].values), freq = 'AS')

elif timescale == 'seasonal_jja':

	month_length = Z.time.dt.days_in_month    
	weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
	Z_weighted = (Z * weights)
	Z = Z_weighted.sel(time=Z_weighted.time.dt.season == 'JJA')
	Z = Z.groupby(Z.time.dt.year).sum("time")
	Z = Z.rename({'year':'time'})            
	Z['time'] = pd.date_range(start=str(Z.time[0].values), end=str(Z.time[-1].values), freq = 'AS')

elif timescale == 'seasonal_son':

	month_length = Z.time.dt.days_in_month    
	weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
	Z_weighted = (Z * weights)
	Z = Z_weighted.sel(time=Z_weighted.time.dt.season == 'SON')
	Z = Z.groupby(Z.time.dt.year).sum("time")
	Z = Z.rename({'year':'time'})            
	Z['time'] = pd.date_range(start=str(Z.time[0].values), end=str(Z.time[-1].values), freq = 'AS')

elif timescale == 'seasonal_djf':

	month_length = Z.time.dt.days_in_month    
	weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
	Z_weighted = (Z * weights)
	Z = Z_weighted.sel(time=Z_weighted.time.dt.season == 'DJF')
	Z = Z.groupby(Z.time.dt.year).sum("time")
	Z = Z.rename({'year':'time'})            
	Z['time'] = pd.date_range(start=str(Z.time[0].values), end=str(Z.time[-1].values), freq = 'AS')
		
# SET: zeros to NaN (impacts means if calculated)

#Z = Z.where(Z > 0.0)

# LOAD: regional mask

mask_3D = regionmask.defined_regions.ar6.land.mask_3D( Z )
n_regions = mask_3D.region.shape[0]

#region_i = mask_3D.sel(region=3)
#Z_i = Z.where(region_i)

# APPLY: latitudinal weights

if use_latitudinal_weighting == True:

	weights = np.cos(np.deg2rad(Z.lat))
	Z_regional_weighted_sum = Z.weighted(mask_3D * weights).sum(dim=("lat", "lon"))

else:
	
	Z_regional_weighted_sum = Z.weighted(mask_3D).sum(dim=("lat", "lon"))

#----------------------------------------------------------------------------
# PLOT: facetgrid of regional timeseries
#----------------------------------------------------------------------------

if plot_facetgrid == True:
    
    figstr = variable + '-' + 'ipcc-ar6-land-region-timeseries' + '-' + 'sum' + '-' + timescale + '.png'
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
# LOOP: over regions
#----------------------------------------------------------------------------

for i in range(n_regions):
    
    print('Region=', i)

    # SMOOTH: with a Gaussian filter (windowed) - to stabilise LOESS fit

    t = Z.time.values
    ts = Z_regional_weighted_sum.isel(region=i).values    
    ts_before_mean = np.nanmean( ts[0:int(nsmooth/2)] )
    ts_after_mean = np.nanmean( ts[-int(nsmooth/2)-1:] )
    ts_windowed = np.hstack([ np.ones( int(nsmooth/2) ) * ts_before_mean, ts, np.ones( int(nsmooth/2) ) * ts_after_mean ])
    ts_smoothed = pd.Series( ts_windowed ).rolling( nsmooth, center=True, win_type='gaussian').mean(std=3).values[int(nsmooth/2):-int(nsmooth/2):]

    # LOESS fit to smooth

    y_loess = sm.nonparametric.lowess( exog=t, endog=ts_smoothed, frac=loess_frac )[:,1]

    # STL fit --> trend
        
    #trend, seasonal, residual = ts_decomposition( ts, freq )

    # OLS fit to raw data

    t_ols, y_ols, slope_ols, intercept_ols = linear_regression_ols( t, ts, method )

    # LTR fit to LOESS

    x_fit, y_fit, corrcoef, R2adj = calculate_piecewise_regression( t, y_loess, depth, nsmooth )

    # EXTRACT: segment breaks

    df_delta_abs = pd.DataFrame( {'delta_abs': np.hstack([ 0, np.round( np.abs( np.diff( y_fit, 2 )), 0 )  ]).ravel()} )
    idx_all = df_delta_abs[df_delta_abs['delta_abs'] > 0].index.to_list()   
    
    '''
    if use_yearly == False:
    
        if len(idx_all) > 0:
            idx_all_diff = np.hstack([ idx_all[0], np.diff( idx_all ) ])
            idx_all_pruned = []
            for k in range(len(idx_all)):
                if idx_all_diff[ k ] > 1:
                    idx_all_pruned.append( idx_all[k] )     
            idx_all = idx_all_pruned
    '''
        
    if len(idx_all) > 1:       
        idx_breaks, slopes_breaks, slopes_all = prune_breakpoints( y_fit, idx_all, slope_threshold)
    else:
        idx_breaks = idx_all

    # EXTRACT: pruned breaks

    idx_pruned = np.setdiff1d( idx_all, idx_breaks )
    idx = [idx_all.index(item) for item in idx_pruned]
    
    idx_breaks = idx_all
       
    if len(idx_breaks) > 2:
    
        # CONSTRAIN: to max = n segments using LTR correlation on the LOESS fit
        
        nsegments = 3
        nbreaks = nsegments - 1
        combinations = list(itertools.combinations(idx_breaks, nbreaks))

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

    #----------------------------------------------------------------------------
    # PLOT: regional timeseries with LOESS, OLS and 3-segement trend lines
    #----------------------------------------------------------------------------

    figstr = variable + '-' + 'ipcc-ar6-land-region-timeseries' + '-' + 'sum' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.png'
    if timescale == 'yearly':
        titlestr = variable + ': aggregated yearly total over IPCC AR6 land region ' + str(i+1)
    elif timescale == 'yearly_jj':
        titlestr = variable + ': aggregated yearly (July-June) total over IPCC AR6 land region ' + str(i+1)
    elif timescale == 'seasonal_mam':
        titlestr = variable + ': aggregated seasonal (MAM) total over IPCC AR6 land region ' + str(i+1)
    elif timescale == 'seasonal_jja':
        titlestr = variable + ': aggregated seasonal (JJA) total over IPCC AR6 land region ' + str(i+1)
    elif timescale == 'seasonal_son':
        titlestr = variable + ': aggregated seasonal (SON) total over IPCC AR6 land region ' + str(i+1)
    elif timescale == 'seasonal_djf':
        titlestr = variable + ': aggregated seasonal (DJF) total over IPCC AR6 land region ' + str(i+1)
    else:
        titlestr = variable + ': aggregated monthly total over IPCC AR6 land region ' + str(i+1)

    if (variable == 'BA_Total') | (variable == 'Cem_Total'):
        ylabelstr = 'Total'
    else:
        ylabelstr = 'TC30_F'

    fig, ax = plt.subplots(figsize=(13.33,7.5))
    plt.plot( t, Z_regional_weighted_sum.isel(region=i).values, color='black', lw=3, alpha = 1, label='Sum' )        

    #plt.plot( t, trend + seasonal + residual, color='red', lw=1, alpha = 1, label='Trend + Seasonal + Residual' )
    #plt.plot( t, trend + seasonal, color='green', lw=1, alpha = 1, label='Trend + Seasonal' )
    #plt.plot( t, trend, color='blue', lw=2, alpha = 1, label='Trend' )
    #plt.plot( t, pd.Series( Z_regional_weighted_sum.isel(region=i).values ).rolling(nsmooth, center=True, win_type='gaussian').mean(std=3), color='red', lw=3, alpha = 1, label='Sum' )

    if timescale == 'yearly':
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='Theil-Sen' )
        plt.plot( t, y_loess, color='blue', lw=3, alpha = 1, label='LOESS' )        
    elif timescale == 'yearly_jj':
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='Theil-Sen' )
        plt.plot( t, y_loess, color='blue', lw=3, alpha = 1, label='LOESS' )        
    elif timescale == 'seasonal_mam':
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='Theil-Sen' )
        plt.plot( t, y_loess, color='blue', lw=3, alpha = 1, label='LOESS' )        
    elif timescale == 'seasonal_jja':
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='Theil-Sen' )
        plt.plot( t, y_loess, color='blue', lw=3, alpha = 1, label='LOESS' )        
    elif timescale == 'seasonal_son':
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='Theil-Sen' )
        plt.plot( t, y_loess, color='blue', lw=3, alpha = 1, label='LOESS' )        
    elif timescale == 'seasonal_djf':
        plt.plot( t, y_ols, color='red', lw=2, alpha = 1, label='Theil-Sen' )
        plt.plot( t, y_loess, color='blue', lw=3, alpha = 1, label='LOESS' )     
        
    if timescale != 'monthly':
        
        if len(idx_pruned) > 0:
            for b in range(len(idx)): 
                if b == 0:
                    plt.axvline( x=t[ idx_all[ idx[b] ] ], lw=1, ls='--', color='red', label='pruned')
                else:
                    plt.axvline( x=t[ idx_all[ idx[b] ] ], lw=1, ls='--', color='red')
            for b in range(len(idx)): plt.text( t[ idx_all[ idx[b] ] ], y_fit[ idx_all[ idx[b] ] ], str( np.round( slopes_all[ idx[b] ], 3 ) ), color='red' )
        if len(idx_breaks) > 0:
            for b in range(len(idx_breaks)): 
                if b == 0:
                    plt.axvline(x=t[idx_breaks[b]], lw=1, ls='--', color='black', label='breakpoint')
                else:
                    plt.axvline(x=t[idx_breaks[b]], lw=1, ls='--', color='black')
            for b in range(len(idx_breaks)): plt.text( t[idx_breaks[b]], y_fit[idx_breaks[b]], str( np.round( slopes_breaks[b], 3 ) ), color='black' )
        plt.plot( t, y_fit, color='yellow', lw=2, alpha = 1, label='LTR segments (n=' + str(len(idx_breaks)+1) + ')' )

                
        # PLOT: OLS segments from LTR (check)
            
        '''
        nbreaks = len(idx_breaks)
        print(i, nbreaks)
        if nbreaks > 0:
                
            for b in range( nbreaks ):                            
                if b == 0:
                    segment_range = np.arange( 0, idx_breaks[b] )                    
                else:                         
                    segment_range = np.arange( idx_breaks[b-1]-1, idx_breaks[b] )
                plt.plot( t[segment_range], y_fit[segment_range], lw=2, color='purple' )
            segment_range = np.arange( idx_breaks[b]-1, len(t) )                    
            plt.plot( t[segment_range], y_fit[segment_range], lw=2, color='purple', label='LTR segments (n=' + str(len(idx_breaks)+1) + ')' )        
        '''

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
