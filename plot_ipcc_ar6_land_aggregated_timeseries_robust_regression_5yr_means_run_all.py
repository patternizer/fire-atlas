#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_ipcc_ar6_land_aggregated_timeseries_robust_regression_5yr_means.py
#------------------------------------------------------------------------------
# Version 0.9
# 29 November, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

# Dataframe libraries:

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

# Plotting libraries:
    
import matplotlib as mpl
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
# %matplotlib inline # for Jupyter Notebooks
#import seaborn as sns; sns.set()
from matplotlib.lines import Line2D
from matplotlib import patches
import matplotlib.image as image

import seaborn as sns

plt.rcParams["font.family"] = "arial"
grey80 = '#808080'
grey84 = '#d6d6d6'
grey90 = '#e5e5e5'
                   
# Stats libraries:

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
import scipy
from scipy import stats
from scipy.stats import t
from scipy.special import erfinv
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from lineartree import LinearTreeClassifier, LinearTreeRegressor

import pwlf
import random
import itertools

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#----------------------------------------------------------------------------
 
#----------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------

def adjusted_r_squared(x,y):
    
    X = x.reshape(len(x),1)
    model = sm.OLS(y, X).fit()
    R2adj = model.rsquared_adj

    return R2adj

def fit_linear_regression( x, y, method, ci ):

    xmean = np.mean(x)
    ymean = np.mean(y) 

    n = len(x)
    X = np.linspace(np.min(x), np.max(x), n)
    X = sm.add_constant(X)
    if method == 'ols':    
        model = sm.OLS(y, X).fit()
    elif method == 'robust':
        model = sm.RLM(y, X).fit() # Theil-Sen

    y_fit = model.predict(X)    
    
    # COMPUTE: number of standard deviations corresponding to c.i.
        
    alpha = 1 - ( ci / 100 )                    # significance level (0.05)
    percentile = 1 - ( alpha / 2 )              # distribution percentile (0.975)
    n_sd = np.sqrt( 2.0 ) * erfinv( ci / 100 )  # 1.96
    
    # COMPUTE: residual standard error    

    residuals = y - y_fit
    dof = n - 2                                 # degrees of freedom: 2 coeffs --> slope and intercept
    n_sd = stats.t.ppf( percentile, dof )       # 1.98
    sse = np.sum( residuals**2 )                # sum of squared residuals
    se = np.sqrt( sse / dof )                   # residual standard error

    '''
    OLS:
        
    beta_1 = np.cov( x, y )[0, 1] / (np.std( x, ddof = 2)**2)
    beta_0 = ymean - beta_1 * xmean
    y_fit = beta_0 + x * beta_1 
    sse = np.sum( ( y - y_fit ) ** 2)

    t_value = beta_1 / (se / np.sqrt( np.sum( (x - xmean)**2 ) ) )

    p_value_lower = t.cdf( -np.abs( t_value ), dof )
    p_value_upper = 1 - t.cdf( t_value, dof )
    p_value = p_value_lower + p_value_upper
    '''
    
    # COMPUTE: uncertainty on the slope
    
    uncertainty = n_sd * se * np.sqrt( 1/n + (x - xmean)**2 / np.sum( (x - xmean)**2 ) )
    lower_bound = y_fit - uncertainty
    upper_bound = y_fit + uncertainty

    # EXTRACT: model parameters and c.i. on parameters
    
    params = model.params    
    params_ci = model.conf_int(alpha=alpha)    
    pvalues = model.pvalues

    return y_fit, lower_bound, upper_bound, params, params_ci, pvalues

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 16
dpi = 300

use_abbrevs = False
use_latitudinal_weighting = False

# LOESS parameters:

loess_frac = 0.6

# OLS regression parameters:

ci = 95
alpha = np.round( 1.0 - ( ci / 100.0 ), 3 )     

# Variable and timescale choice:

variable_list = [ 'BA_Total', 'BA_Forest_NonForest', 'Cem_Total', 'Cem_Forest_NonForest' ]
timescale_list = [ 'yearly', 'yearly_jj', 'seasonal_mam', 'seasonal_jja', 'seasonal_son', 'seasonal_djf', 'monthly' ]

#variable = variable_list[0]
#timescale = timescale_list[0]

#----------------------------------------------------------------------------
# RUN:
#----------------------------------------------------------------------------

for variable in variable_list:   
    for timescale in timescale_list:

        nc_file = 'OUT/' + variable + '.nc'
        
        if variable == 'BA_Total':
            variablestr = 'Burned Area'
            unitstr = '(thousand km' + r'$^{2}$' + ')'
        elif variable == 'BA_Forest_NonForest':
            variablestr = 'Burned Area (in forest)'
            unitstr = '(thousand km' + r'$^{2}$' + ')'
        elif variable == 'Cem_Total':
            variablestr = 'Carbon Emissions'
            unitstr = '(thousand tonnes C)'
        elif variable == 'Cem_Forest_NonForest':
            variablestr = 'Carbon Emissions (in forest)'
            unitstr = '(thousand tonnes C)'
        
        if timescale == 'yearly':
        
        	nsmooth = 5
        	use_all_but_last_year = False
        	method = 'theil_sen'
        	timescalestr = 'Annual (JAN-DEC)'
        	
        elif timescale == 'yearly_jj':
        
        	nsmooth = 5
        	use_all_but_last_year = False
        	method = 'theil_sen'
        	timescalestr = 'Annual (JUL-JUN)'
        
        elif timescale == 'seasonal_mam':
        
        	nsmooth = 5
        	use_all_but_last_year = False
        	method = 'theil_sen'    
        	timescalestr = 'Seasonal (MAM)'
        
        elif timescale == 'seasonal_jja':
        
        	nsmooth = 5
        	use_all_but_last_year = False
        	method = 'theil_sen'    
        	timescalestr = 'Seasonal (JJA)'
        
        elif timescale == 'seasonal_son':
        
        	nsmooth = 5
        	use_all_but_last_year = False
        	method = 'theil_sen'    
        	timescalestr = 'Seasonal (SON)'
        
        elif timescale == 'seasonal_djf':
        
        	nsmooth = 5
        	use_all_but_last_year = False
        	method = 'theil_sen'    
        	timescalestr = 'Seasonal (DJF)'
        
        elif timescale == 'monthly': 
        
        	nsmooth = 60
        	use_all_but_last_year = False
        	method = 'ols' # Theil-Sen fails to converge for "flat" LOESS in monthly series
        	timescalestr = 'Monthly'
        
        titlestr_ = timescalestr + ' ' + variablestr.lower()
        ylabelstr = variablestr[:1] + variablestr.lower()[1:] + ' ' + unitstr
        
        #----------------------------------------------------------------------------
        # LOAD: netCDF4
        #----------------------------------------------------------------------------
        
        ds = xr.load_dataset( nc_file )
        
        #----------------------------------------------------------------------------
        # EXTRACT: regional mean timeseries and compute statistics
        #----------------------------------------------------------------------------
        
        Z = ds[variable]
        
        if (variable == 'BA_Total') | (variable == 'BA_Forest_NonForest'):
            
            Z = Z / 1000.0
            
        elif (variable == 'Cem_Total') | (variable == 'Cem_Forest_NonForest'):
            
            Z = Z / 1.0e9
        
        if timescale == 'yearly':
        
        	Z = Z.resample(time="AS").sum(dim="time")   
        
        	if use_all_but_last_year == True:    
        		
        		Z = Z.sel(time=slice(Z.time[0],Z.time[-2]))
        
        elif timescale == 'yearly_jj':
        
        	Z = Z.resample(time='AS-JUL').sum('time') # anchored offset for the austral year
        
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
        
        # APPLY: latitudinal weights
        
        if use_latitudinal_weighting == True:
            
        	weights = np.cos(np.deg2rad(Z.lat))
        	Z_regional_sum = Z.weighted(mask_3D * weights).sum(dim=("lat", "lon"))
        
        else:
        	
        	Z_regional_sum = Z.weighted(mask_3D).sum(dim=("lat", "lon"))
        
        #----------------------------------------------------------------------------
        # LOOP: over regions
        #----------------------------------------------------------------------------
        
        for i in range(n_regions):
            
            print('Region=', i)
        
            # EXTRACT: regional timeseries    
        
            t = Z.time.values
            ts = Z_regional_sum.isel(region=i).values    
        
            ''' 
            # APPLY: year mask
        
            if timescale == 'yearly':
        
                # YEAR MASK: incomplete years
                        
                t = Z.time.values
                years = np.array( [ pd.Series(t)[i].year for i in range(len(t)) ] )
                year_mask = []
                for year in np.unique(years):
                    if len( years[years == year] ) == 12:
                        year_mask.append( True )
                    else:
                        year_mask.append( False )
                
                t = t[ year_mask ]
                ts = ts[ year_mask ]
            '''
                
            tstart_ols = t[0]
            tend_ols = t[-1]
        
            # INITIALISE: stat vectors
        
            fittypes = []
            slopes = []
            intercepts = []
            corrcoefs = []
            R2adjs = []
            tstarts = []
            tends = []
        
            if np.subtract( ts, np.zeros(len(ts)) ).sum() == 0:
                
                no_data = True
        
                slope_ols = np.nan 
                intercept_ols = np.nan 
                corrcoef_ols = np.nan 
                R2adj_ols = np.nan 
                
            else:
        
                no_data = False
        
                # SMOOTH: with a Gaussian filter (windowed) - to stabilise LOESS fit
                
                ts_before_mean = np.nanmean( ts[0:int(nsmooth/2)] )
                ts_after_mean = np.nanmean( ts[-int(nsmooth/2)-1:] )
                ts_windowed = np.hstack([ np.ones( int(nsmooth/2) ) * ts_before_mean, ts, np.ones( int(nsmooth/2) ) * ts_after_mean ])
                ts_smoothed = pd.Series( ts_windowed ).rolling( nsmooth, center=True, win_type='gaussian').mean(std=3).values[int(nsmooth/2):-int(nsmooth/2):]
            
                # LOESS fit to smooth
            
                y_loess = sm.nonparametric.lowess( exog=t, endog=ts_smoothed, frac=loess_frac )[:,1]
            
                # robust OLS fit to LOESS + goodness of fit stats
            
                t_idx = np.linspace( 0, len(t)-1, num=len(t) )
                y_ols, lower_bound_ols, upper_bound_ols, params_ols, params_ci_ols, pvalues_ols = fit_linear_regression( t_idx, ts, 'robust', ci )
            
                intercept_ols = params_ols[0]
                slope_ols = params_ols[1]
            
                hypothesis_test_ols = int( pvalues_ols[1] < alpha )
                if hypothesis_test_ols == 0:
                    hypothesis_teststr = r'$H_{0}$'
                else:
                    hypothesis_teststr = r'$H_{A}$'
            
                corrcoef_ols = scipy.stats.pearsonr(y_loess, y_ols)[0]
                R2adj_ols = adjusted_r_squared(y_loess, y_ols)
                
            fittypes.append( 'theil-sen' )
            slopes.append( slope_ols )
            intercepts.append( intercept_ols )
            corrcoefs.append( corrcoef_ols )
            R2adjs.append( R2adj_ols )
            tstarts.append( tstart_ols )
            tends.append( tend_ols )
                
            #----------------------------------------------------------------------------
            # 5-yr rolling means
            #----------------------------------------------------------------------------
            
            #y = ts.isel(region=i).values
            y = ts
            s = pd.Series( y, index = t )    
            if timescale == 'yearly':
                
                df_intervals = pd.DataFrame([
                    ['2000-01-01', '2004-12-31'],
                    ['2005-01-01', '2009-12-31'],
                    ['2010-01-01', '2014-12-31'],
                    ['2015-01-01', '2019-12-31'],
                    ['2020-01-01', '2024-12-31'],        
                ], columns=['StartDate', 'EndDate'], dtype='datetime64[ns]')
                df_intervals['5yr-mean'] = df_intervals.apply(lambda row: s[(row['StartDate'] <= s.index) & (s.index < row['EndDate'])].mean(), axis=1)
        
            elif timescale == 'yearly_jj':
        
                df_intervals = pd.DataFrame([
                    ['2000-07-01', '2005-06-30'],
                    ['2005-07-01', '2010-06-30'],
                    ['2010-07-01', '2015-06-30'],
                    ['2015-07-01', '2020-06-30'],
                    ['2020-07-01', '2025-06-30'],        
                ], columns=['StartDate', 'EndDate'], dtype='datetime64[ns]')
                df_intervals['5yr-mean'] = df_intervals.apply(lambda row: s[(row['StartDate'] <= s.index) & (s.index < row['EndDate'])].mean(), axis=1)
        
            elif timescale == 'seasonal_djf':
        
                df_intervals = pd.DataFrame([
                    ['2000-01-01', '2004-12-31'],
                    ['2005-01-01', '2009-12-31'],
                    ['2010-01-01', '2014-12-31'],
                    ['2015-01-01', '2019-12-31'],
                    ['2020-01-01', '2024-12-31'],        
                ], columns=['StartDate', 'EndDate'], dtype='datetime64[ns]')
                df_intervals['5yr-mean'] = df_intervals.apply(lambda row: s[(row['StartDate'] <= s.index) & (s.index < row['EndDate'])].mean(), axis=1)
        
            elif timescale == 'seasonal_mam':
        
                df_intervals = pd.DataFrame([
                    ['2000-01-01', '2004-12-31'],
                    ['2005-01-01', '2009-12-31'],
                    ['2010-01-01', '2014-12-31'],
                    ['2015-01-01', '2019-12-31'],
                    ['2020-01-01', '2024-12-31'],        
                ], columns=['StartDate', 'EndDate'], dtype='datetime64[ns]')
                df_intervals['5yr-mean'] = df_intervals.apply(lambda row: s[(row['StartDate'] <= s.index) & (s.index < row['EndDate'])].mean(), axis=1)
        
            elif timescale == 'seasonal_jja':
        
                df_intervals = pd.DataFrame([
                    ['2000-01-01', '2004-12-31'],
                    ['2005-01-01', '2009-12-31'],
                    ['2010-01-01', '2014-12-31'],
                    ['2015-01-01', '2019-12-31'],
                    ['2020-01-01', '2024-12-31'],        
                ], columns=['StartDate', 'EndDate'], dtype='datetime64[ns]')
                df_intervals['5yr-mean'] = df_intervals.apply(lambda row: s[(row['StartDate'] <= s.index) & (s.index < row['EndDate'])].mean(), axis=1)
        
            elif timescale == 'seasonal_son':
        
                df_intervals = pd.DataFrame([
                    ['2000-01-01', '2004-12-31'],
                    ['2005-01-01', '2009-12-31'],
                    ['2010-01-01', '2014-12-31'],
                    ['2015-01-01', '2019-12-31'],
                    ['2020-01-01', '2024-12-31'],        
                ], columns=['StartDate', 'EndDate'], dtype='datetime64[ns]')
                df_intervals['5yr-mean'] = df_intervals.apply(lambda row: s[(row['StartDate'] <= s.index) & (s.index < row['EndDate'])].mean(), axis=1)
               
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
                df_timeseries.to_csv( 'RUN/' + variable + '-' + 'timeseries' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.csv', index = False )
        		
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
                df_timeseries.to_csv( 'RUN/' + variable + '-' + 'timeseries' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.csv', index = False )
        		
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
            df_stats.to_csv( 'RUN/' + variable + '-' + 'stats' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.csv', index = False )
        
            #----------------------------------------------------------------------------
            # PLOT: regional timeseries with LOESS, OLS and 3-segement trend lines
            #----------------------------------------------------------------------------
        
            figstr = variable + '-' + 'ipcc-ar6-land-region-timeseries' + '-' + 'sum' + '-' + timescale + '-' + 'region' + '-' + str(i+1).zfill(2) + '.png'
        
            fig, ax = plt.subplots(figsize=(13.33,7.5))
        
            if no_data == True:
            
                plt.plot( t, y, marker=None, ls='None' )        
                if timescale != 'monthly':
                    plt.text( t[ (int(len(t)/2)-1) ], 0.0, 'NO DATA', fontsize=fontsize )
                else:
                    plt.text( t[ (int(len(t)/2)-12) ], 0.0, 'NO DATA', fontsize=fontsize )
                plt.yticks([])
                
            elif no_data == False:
                        
                if timescale != 'monthly':        
            
                    # PLOT: timeseries
            
                    plt.plot( t, y, 'o', markerfacecolor='lightgrey', ls='-', color='black', lw=2, alpha = 1, label=variablestr, zorder=10 )        
            
                    # PLOT: 5-yr interval means
                    
                    tdelta = int(365/2)                    
                    for k in range(len(df_intervals)):
                        
                        if (k < len(df_intervals) - 1):
                            t_interval = t[ ( t>=df_intervals.StartDate[k] ) & ( t<=df_intervals.StartDate[k+1] ) ] - np.timedelta64( tdelta, 'D')
                        else: 
                            t_interval = t[ ( t>=df_intervals.StartDate[k] ) & ( t<=df_intervals.EndDate[k] ) ] - np.timedelta64( tdelta, 'D')
                            if len(t_interval) > 1:
                                t_interval[-1] = t_interval[-1] + np.timedelta64(365,'D')
                            else:
                                t_interval = np.hstack( [ t_interval, t_interval[-1] + np.timedelta64(365,'D') ] )                
                    
                        y_interval = [ df_intervals['5yr-mean'][k] ] * len( t_interval )
                        if k == 0:
                            #plt.plot( t_interval, y_interval, solid_capstyle='butt', color='cyan', lw=10, alpha=0.5, label='5-yr mean', zorder=4 )
                            plt.plot( t_interval, y_interval, solid_capstyle='butt', color='cyan', lw=10, alpha=0.5, label='Half-decade average', zorder=4 )
                        else:
                            plt.plot( t_interval, y_interval, solid_capstyle='butt', color='cyan', lw=10, alpha=0.5, zorder=4 )
                
                    # PLOT: robust OLS uncertainty band + p-value
                
                    if pvalues_ols[1] < 0.001:            
                        pvaluestr = 'p<0.001'
                    else:
                        pvaluestr = 'p=' + str( np.round( pvalues_ols[1], 3 ) )
                
                    if pvalues_ols[1] <= alpha:
                        #plt.plot(t, y_ols, color='red', lw=3, label='Theil-Sen (' + r'$\beta_{0}=$' + str( (np.round( params_ols[0], 1)) ) + ', ' + r'$\beta_{1}$=' + str( (np.round( params_ols[1], 1 )) ) + r' $yr^{-1}$)', zorder=5)                            
                        #plt.fill_between(t, lower_bound_ols, upper_bound_ols, color='red', alpha=0.1, label='Theil-Sen 95% c.i. (' + pvaluestr  + ', ' + hypothesis_teststr + ')', zorder=1)
                        plt.plot(t, y_ols, color='red', lw=3, label='Trend (if p < 0.05)', zorder=5)                            
                        plt.fill_between(t, lower_bound_ols, upper_bound_ols, color='red', alpha=0.1, label='95% confidence level', zorder=1)
        
                    legend_elements = [
                        Line2D( [0], [0], marker='o', markerfacecolor='lightgrey', ls='-', color='black', lw=2, alpha = 1, label=variablestr),
                        Line2D( [0], [1], solid_capstyle='butt', color='cyan', lw=10, alpha=0.5, label='Half-decade average'),
                        Line2D( [1], [0], color='red', lw=3, label='Trend (if p < 0.05)'),
                        patches.Patch( [1], [1], color='red', alpha=0.1, label='95% confidence level')                                
                    ]    
                    plt.legend( handles = legend_elements, ncol=2, labelcolor = grey80, fontsize = fontsize, loc='center', bbox_to_anchor=(0.7, -0.2), fancybox = False, shadow = False, frameon=True )
                        
                else:
                    
                    # PLOT: monthly timeseries
            
                    plt.plot( t, y, 'o', markersize=3, markerfacecolor='lightgrey', ls='-', color='black', lw=2, alpha = 1)                
                    
            fig.autofmt_xdate(rotation=0, ha='center')
            
            ax.xaxis.set_major_formatter( mdates.DateFormatter('%Y') )
            #ax.fmt_xdata = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            #ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            #ax.grid( True )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_tick_params(direction='out', width=1)
            ax.get_yaxis().set_tick_params(direction='out', width=1)
            ax = plt.gca()
            #ax.grid( which='minor', axis='y', linestyle='-', color=grey80, alpha=1 )
            ax.grid( which='major', axis='y', linestyle='-', color=grey90, alpha=1 )    
            plt.xlim( [np.datetime64('2000'), np.datetime64('2025')] )
            #plt.ylim( [np.min(y), np.max(y)] )
            plt.tick_params( colors=grey80, labelsize=fontsize )    
        
            sns.despine( offset=10, trim=True)
        
            plt.ylabel( ylabelstr, color=grey80, fontsize=fontsize )
            if use_abbrevs == True:
                titlestr = titlestr_ + ' (' + mask_3D.abbrevs.values[i] + ')'
            else:
                titlestr = titlestr_ + ' (' + mask_3D.region.names.values[i] + ')'
            plt.title( titlestr, color=grey80, fontsize=fontsize )    
        
            # CREDITS:    
        
            plt.annotate( 'Data: Jones et al (2022)\ndoi: 10.1029/2020RG000726\nDataViz: Michael Taylor', xy=(280,80), xycoords='figure pixels', color = grey84, fontsize = fontsize )   
        
            # LOGO:    
                
            im = image.imread('logo-cru.png')
            imax = fig.add_axes([0.36, 0.06, 0.05, 0.05])
            imax.set_axis_off()
            imax.imshow(im, aspect="equal")
        
            plt.savefig( figstr, dpi=dpi, bbox_inches='tight' )
            plt.close()

#----------------------------------------------------------------------------
print('** END')
