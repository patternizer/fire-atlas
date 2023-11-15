#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_ipcc_ar6_land_aggregated_timeseries_3segment.py
#------------------------------------------------------------------------------
# Version 0.5
# 15 November, 2023
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

    mask = np.isfinite(y_obs) & np.isfinite(y_fit.reshape(-1,1))
    corrcoef_fit = scipy.stats.pearsonr(y_obs[mask], y_fit.reshape(-1,1)[mask])[0]
    R2adj_fit = adjusted_r_squared(y_obs, y_fit.reshape(-1,1))
    
    return x_fit, y_fit, corrcoef_fit, R2adj_fit

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
dpi = 300

use_abbrevs = True
use_latitudinal_weighting = False
plot_facetgrid = True

# LOESS parameters:

loess_frac = 0.6
    
# LTR parameters:

depth = 5
nbreaks_max = 2

# OLS regression parameters:
    
#method = 'ols'    
#method = 'theil_sen'    
#method = 'ransac'    

variable_list = [ 'BA_Total', 'BA_Forest_NonForest', 'Cem_Total', 'Cem_Forest_NonForest' ]
timescale_list = [ 'yearly', 'yearly_jj', 'seasonal_mam', 'seasonal_jja', 'seasonal_son', 'seasonal_djf', 'monthly' ]

#----------------------------------------------------------------------------
# RUN:
#----------------------------------------------------------------------------

for variable in variable_list:
    
    for timescale in timescale_list:
        
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
                
        # SET: zeros to NaN (impacts means)
        
        #Z = Z.where(Z > 0.0)
        
        mask_3D = regionmask.defined_regions.ar6.land.mask_3D( Z )
        n_regions = mask_3D.region.shape[0]
        
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
        
            fittypes = []
            slopes = []
            intercepts = []
            corrcoefs = []
            R2adjs = []
            tstarts = []
            tends = []
            
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
        
            # OLS fit to LOESS + compute goodness of fit stats
        
            t_ols, y_ols, slope_ols, intercept_ols = linear_regression_ols( t, ts, method )
        
            corrcoef_ols = scipy.stats.pearsonr(y_loess, y_ols)[0]
            R2adj_ols = adjusted_r_squared(y_loess, y_ols)
            tstart_ols = t_ols[0]
            tend_ols = t_ols[-1]
        
            fittypes.append( 'theil-sen' )
            slopes.append( slope_ols )
            intercepts.append( intercept_ols )
            corrcoefs.append( corrcoef_ols )
            R2adjs.append( R2adj_ols )
            tstarts.append( tstart_ols )
            tends.append( tend_ols )
        
            # LTR fit to LOESS
        
            x_fit, y_fit, corrcoef_fit, R2adj_fit = calculate_piecewise_regression( t, y_loess, depth, nsmooth )
        
            slope_fit = np.nan
            intercept_fit = np.nan
            tstart_fit = t[0]
            tend_fit = t[-1]
        
            fittypes.append( 'ltr_fit' )
            slopes.append( slope_fit )
            intercepts.append( intercept_fit )
            corrcoefs.append( corrcoef_fit )
            R2adjs.append( R2adj_fit )
            tstarts.append( tstart_fit )
            tends.append( tend_fit )
        
            #----------------------------------------------------------------------------
            # EXTRACT: segment breaks + compute goodness of fit stats per segement & overall fit
            #----------------------------------------------------------------------------
        
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
        
                # HANDLE: case of 2 breaks at neighbouring timestamps
                
                if len(idx_breaks) == 2:
                    
                    if np.diff( idx_breaks ) == 1:
                        
                        idx_breaks = np.array( [idx_breaks[0]] )
        
                # FIT: optimum curve
                    
                y_best = []
                for k in range(len(idx_breaks)):            
                    
                    if k == 0:
        
                        t_segment = t[0:idx_breaks[k]]
                        ts_segment = y_loess[0:idx_breaks[k]]
                        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                        y_best.append(ts_segment_ols)
        
                        corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                        R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                        tstart_segment = t_segment[0]
                        tend_segment = t_segment[-1]   
                        
                        fittypes.append( 'ltr_segment' )
                        slopes.append( slope_segment_ols )
                        intercepts.append( intercept_segment_ols )
                        corrcoefs.append( corrcoef_segment )
                        R2adjs.append( R2adj_segment )
                        tstarts.append( tstart_segment )
                        tends.append( tend_segment )
        
                    else: 
        
                        t_segment = t[idx_breaks[k-1]:idx_breaks[k]]
                        ts_segment = y_loess[idx_breaks[k-1]:idx_breaks[k]]                
                        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                        y_best.append(ts_segment_ols)    
        
                        corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                        R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                        tstart_segment = t_segment[0]
                        tend_segment = t_segment[-1]   
                        
                        fittypes.append( 'ltr_segment' )
                        slopes.append( slope_segment_ols )
                        intercepts.append( intercept_segment_ols )
                        corrcoefs.append( corrcoef_segment )
                        R2adjs.append( R2adj_segment )
                        tstarts.append( tstart_segment )
                        tends.append( tend_segment )
        
                t_segment = t[idx_breaks[k]:]
                ts_segment = y_loess[idx_breaks[k]:]
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)
        
                corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                tstart_segment = t_segment[0]
                tend_segment = t_segment[-1]   
                    
                fittypes.append( 'ltr_segment' )
                slopes.append( slope_segment_ols )
                intercepts.append( intercept_segment_ols )
                corrcoefs.append( corrcoef_segment )
                R2adjs.append( R2adj_segment )
                tstarts.append( tstart_segment )
                tends.append( tend_segment )
        
                # JOIN: segments
                        
                y_best = np.hstack( y_best )        
            
            elif len(idx_breaks) == 2:
        
                # HANDLE: case of 2 breaks at neighbouring timestamps
                
                if np.diff( idx_breaks ) == 1: idx_breaks = np.array( [idx_breaks[0]] )

                # FIT: optimum curve
                
                y_best = []
                for k in range(len(idx_breaks)):            
        
                    if k == 0:
        
                        t_segment = t[0:idx_breaks[k]]
                        ts_segment = y_loess[0:idx_breaks[k]]
                        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                        y_best.append(ts_segment_ols)
        
                        corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                        R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                        tstart_segment = t_segment[0]
                        tend_segment = t_segment[-1]   
                        
                        fittypes.append( 'ltr_segment' )
                        slopes.append( slope_segment_ols )
                        intercepts.append( intercept_segment_ols )
                        corrcoefs.append( corrcoef_segment )
                        R2adjs.append( R2adj_segment )
                        tstarts.append( tstart_segment )
                        tends.append( tend_segment )
        
                    else: 
        
                        t_segment = t[idx_breaks[k-1]:idx_breaks[k]]
                        ts_segment = y_loess[idx_breaks[k-1]:idx_breaks[k]]                
                        t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                        y_best.append(ts_segment_ols)    
        
                        corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                        R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                        tstart_segment = t_segment[0]
                        tend_segment = t_segment[-1]   
                        
                        fittypes.append( 'ltr_segment' )
                        slopes.append( slope_segment_ols )
                        intercepts.append( intercept_segment_ols )
                        corrcoefs.append( corrcoef_segment )
                        R2adjs.append( R2adj_segment )
                        tstarts.append( tstart_segment )
                        tends.append( tend_segment )
                        
                    # STORE: segment stats
            
                    corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                    R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                    tstart_segment = t_segment[0]
                    tend_segment = t_segment[-1]   
                    
                    fittypes.append( 'ltr_segment' )
                    slopes.append( slope_segment_ols )
                    intercepts.append( intercept_segment_ols )
                    corrcoefs.append( corrcoef_segment )
                    R2adjs.append( R2adj_segment )
                    tstarts.append( tstart_segment )
                    tends.append( tend_segment )
                                        
                t_segment = t[idx_breaks[k]:]
                ts_segment = y_loess[idx_breaks[k]:]
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)
        
                # STORE: segment stats
        
                corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                tstart_segment = t_segment[0]
                tend_segment = t_segment[-1]        
        
                fittypes.append( 'ltr_segment')
                slopes.append( slope_segment_ols )
                intercepts.append( intercept_segment_ols )
                corrcoefs.append( corrcoef_segment )
                R2adjs.append( R2adj_segment )
                tstarts.append( tstart_segment )
                tends.append( tend_segment )
        
                # JOIN: segments
                        
                y_best = np.hstack( y_best )        
                
            elif len(idx_breaks) == 1:
                    
                k = 0
                
                # FIT: optimum curve
                
                y_best = []
                t_segment = t[0:idx_breaks[k]]
                ts_segment = y_loess[0:idx_breaks[k]]
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)
        
                # STORE: segment stats
        
                corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                tstart_segment = t_segment[0]
                tend_segment = t_segment[-1]        
        
                fittypes.append( 'ltr_segment')
                slopes.append( slope_segment_ols )
                intercepts.append( intercept_segment_ols )
                corrcoefs.append( corrcoef_segment )
                R2adjs.append( R2adj_segment )
                tstarts.append( tstart_segment )
                tends.append( tend_segment )
        
                t_segment = t[idx_breaks[k]:]
                ts_segment = y_loess[idx_breaks[k]:]
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best.append(ts_segment_ols)
        
                corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                tstart_segment = t_segment[0]
                tend_segment = t_segment[-1]        
        
                fittypes.append( 'ltr_segment')
                slopes.append( slope_segment_ols )
                intercepts.append( intercept_segment_ols )
                corrcoefs.append( corrcoef_segment )
                R2adjs.append( R2adj_segment )
                tstarts.append( tstart_segment )
                tends.append( tend_segment )
        
                # JOIN: segments
                        
                y_best = np.hstack( y_best )        
        
            elif len(idx_breaks) == 0:
                
                # FIT: optimum curve
                
                t_segment = t
                ts_segment = y_loess
                t_segment_ols, ts_segment_ols, slope_segment_ols, intercept_segment_ols = linear_regression_ols( t_segment, ts_segment, method )
                y_best = ts_segment_ols
        
                # STORE: segment stats
        
                corrcoef_segment = scipy.stats.pearsonr(ts_segment, ts_segment_ols)[0]
                R2adj_segment = adjusted_r_squared(ts_segment, ts_segment_ols)
                tstart_segment = t_segment[0]
                tend_segment = t_segment[-1]
        
                fittypes.append( 'ltr_segment')
                slopes.append( slope_segment_ols )
                intercepts.append( intercept_segment_ols )
                corrcoefs.append( corrcoef_segment )
                R2adjs.append( R2adj_segment )
                tstarts.append( tstart_segment )
                tends.append( tend_segment )
        
            # COMPUTE: goodness of fit for 3-segment fit
        
            slope_best = np.nan
            intercept_best = np.nan
            corrcoef_best = scipy.stats.pearsonr(y_loess, y_best)[0]
            R2adj_best = adjusted_r_squared(y_loess, y_best)
            tstart_best = t[0]
            tend_best = t[-1]
        
            fittypes.append( 'ltr_3_segment' )
            slopes.append( slope_best )
            intercepts.append( intercept_best )
            corrcoefs.append( corrcoef_best )
            R2adjs.append( R2adj_best )
            tstarts.append( tstart_best )
            tends.append( tend_best )
        
            #----------------------------------------------------------------------------
            # SAVE: regional timeseries to CSV
            #----------------------------------------------------------------------------
            
            if timescale != 'monthly':
        
                years = np.array( [ pd.Series(t)[i].year for i in range(len(t)) ] )
                df_timeseries = pd.DataFrame(columns=['year', 'sum', 'variable', 'timescale', 'region'])
                df_timeseries['year'] = years
                df_timeseries['sum'] = ts
                df_timeseries['variable'] = [variable] * len(years)
                df_timeseries['timescale'] = [timescale] * len(years)
                df_timeseries['region'] = ['region' + '_' + str(i+1).zfill(2)] * len(years)    
                df_timeseries.to_csv( variable + '-' + 'timeseries' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.csv' )
                
            else:
        
                years = np.unique( np.array( [ pd.Series(t)[i].year for i in range(len(t)) ] ) )
                datetimes_data = np.array( [ pd.Series(t)[i] for i in range(len(t)) ] )
        
                # PAD: timeseries to complete years
                
                datetimes_filled = pd.date_range( start = str(years[0]), end = str(years[-1]+1), freq='MS')[0:-1]
                df_filled = pd.DataFrame({'datetime':datetimes_filled})
                df_data = pd.DataFrame({'datetime':datetimes_data, 'ts':ts})              
                df = df_filled.merge(df_data, how='left', on='datetime')
                ts_filled = df.ts.values
                ts_filled_array = ts_filled.reshape( [len(years), 12] )
        
                df_timeseries = pd.DataFrame(columns=['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'variable', 'timescale', 'region'])
                df_timeseries['year'] = years        
                for m in range(12):
                    df_timeseries[str(m+1)] = ts_filled_array[:,m]
                df_timeseries['variable'] = [variable] * len(years)
                df_timeseries['timescale'] = [timescale] * len(years)
                df_timeseries['region'] = ['region' + '_' + str(i+1).zfill(2)] * len(years)    
                df_timeseries.to_csv( variable + '-' + 'timeseries' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.csv' )
                
            #----------------------------------------------------------------------------
            # SAVE: regional trend stats to CSV
            #----------------------------------------------------------------------------
        
            df_stats = pd.DataFrame(columns=['fittype', 'tstart', 'tend', 'slope', 'intercept', 'corrcoef', 'R2adj', 'variable', 'timescale', 'region'])       
            df_stats['fittype'] = fittypes
            df_stats['tstart'] = tstarts
            df_stats['tend'] = tends
            df_stats['slope'] = slopes
            df_stats['intercept'] = intercepts
            df_stats['corrcoef'] = corrcoefs
            df_stats['R2adj'] = R2adjs            
            df_stats['variable'] = [variable] * len(fittypes)
            df_stats['timescale'] = [timescale] * len(fittypes)
            df_stats['region'] = ['region' + '_' + str(i+1).zfill(2)] * len(fittypes)
            df_stats = df_stats.round(decimals=6)    
            df_stats.to_csv( variable + '-' + 'stats' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.csv' )
            
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
