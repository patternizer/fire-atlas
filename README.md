![image](https://github.com/patternizer/fire-atlas/blob/main/PLOTS/BA_Qdeg_C6_202306.png)

# fire-atlas
UEA impact study to generate a global wildfire atlas with FireCrew

## Contents

* `plot_hdf5.py` - Python script to extract an HDF5 variable and plot it on a world map using Cartopy
* `hdf5_2_netcdf.py` - Python script to extract HDF5 variable arrays per timestep and save as netCDF4
* `merge_netcdfs.py` - Python script to concatenate extracted netCDF4 files per variable arrays per timestep
* `plot_timeseries.py` - Python script to find maximum variance gridcell per variable and plot extracted timeseries

The first step is to clone the latest fire-atlas repo and step into the check out directory: 

    $ git clone https://github.com/patternizer/fire-atlas.git
    $ cd fire-atlas

### Usage

The code was tested locally in a Python 3.8.16 virtual environment.

    $ python plot_hdf5.py (optional test plot for a single variable timestamp)
    $ python hdf5_2_netcdf.py
    $ python merge_netcdfs.py
    $ python plot_timeseries.py
        
## License

TBC ( currently Unlicence )

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)
