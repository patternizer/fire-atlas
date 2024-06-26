![image](https://github.com/patternizer/fire-atlas/blob/main/PLOTS/BA_Qdeg_C6_202306.png)

# fire-atlas
UEA impact study to generate a global wildfire atlas with FireCrew

## Contents

* `plot_hdf5.py` - Python script to extract an HDF5 variable and plot a timestamp on a world map using Cartopy
* `plot_hdf5.sh` - SLURM bash script submit plot_hdf5.py to the ADA job submission queue
* `hdf5_2_netcdf.py` - Python script to extract HDF5 variable arrays per timestep and save as netCDF4
* `merge_netcdfs.py` - Python script to concatenate extracted netCDF4 files per variable arrays per timestep
* `convert_netcdf_SN_2_NS.py` - Python script to flip satellite read row order for latitudes from SN --> NS
* `plot_highest_variance_gridcell_timeseries.py` - Python script to find maximum variance gridcell per variable and plot extracted timeseries and variable mean map + location
* `plot_ipcc_ar6_region_classifications.py` - Python script to extract IPCC AR6 region masks and plot classification on world map
* `plot_ipcc_ar6_land_aggregated_timeseries_stats.py` - Python script to compute gridcell stats for each monthly and yearly sampled variable per IPCC AR6 region
* `plot_ipcc_ar6_land_aggregated_timeseries_pruning.py` - Python script to aggregrate total variable monthly and yearly sampled timeseries per IPCC AR6 region and then segment the LOESS trend by optimisation of breakpoints
* `plot_ipcc_ar6_land_aggregated_timeseries_robust_regression_5yr_means.py` - Python script to aggregrate total variable monthly and yearly sampled timeseries per IPCC AR6 region and then perform a regression t-test and plot robust OLS fit + uncertainty band with 5-yr means
* `create_ipcc_ar6_region_svg.py` - Python script to proudce a SVG world map for IPCC AR6 land regions

The first step is to clone the latest fire-atlas repo and step into the check out directory: 

    $ git clone https://github.com/patternizer/fire-atlas.git
    $ cd fire-atlas

### Usage

The code was tested locally in a Python 3.8.16 virtual environment.

    $ python plot_hdf5.py (optional)
    $ python plot_ipcc_ar6_region_classifications.py (optional)
    $ python hdf5_2_netcdf.py
    $ python merge_netcdfs.py
    $ python convert_netcdf_SN_2_NS.py
    $ python plot_ipcc_ar6_land_aggregated_timeseries_stats.py (optional)
    $ python plot_ipcc_ar6_land_aggregated_timeseries_pruning.py (optional)
	$ python plot_ipcc_ar6_land_aggregated_timeseries_robust_regression_5yr_means.py
    $ python plot_ipcc_ar6_land_region_svg.py
        
## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)
