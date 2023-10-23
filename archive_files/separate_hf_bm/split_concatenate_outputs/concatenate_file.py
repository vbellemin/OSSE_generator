############ IMPORTS ############

import os

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xskillscore
from scipy.interpolate import griddata
from scipy import signal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

import scipy.fftpack as fp

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

import gc 



date = np.arange(np.datetime64("2012-10-29"),np.datetime64("2012-10-31"))

for d in date : 
    date_str = d.astype('str').replace("-","")

    #FILE IMPORT#
    ds1 = xr.open_mfdataset(["/bettik/bellemva/MITgcm/MITgcm_filtered_second/split_10.0_180.0/"+date_str+".nc",
                         "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split_10.0_200.0/"+date_str+".nc",
                         "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split_10.0_220.0/"+date_str+".nc"])

    ds2 = xr.open_mfdataset(["/bettik/bellemva/MITgcm/MITgcm_filtered_second/split30.0_180.0/"+date_str+".nc",
                            "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split30.0_190.0/"+date_str+".nc",
                            "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split30.0_200.0/"+date_str+".nc",
                            "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split30.0_210.0/"+date_str+".nc",
                            "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split30.0_220.0/"+date_str+".nc",
                            "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split30.0_230.0/"+date_str+".nc"])

    ds3 = xr.open_mfdataset(["/bettik/bellemva/MITgcm/MITgcm_filtered_second/split40.0_180.0/"+date_str+".nc",
                        "/bettik/bellemva/MITgcm/MITgcm_filtered_second/split40.0_205.0/"+date_str+".nc"])

    ds4 = xr.open_dataset("/bettik/bellemva/MITgcm/MITgcm_filtered_second/split240.0/"+date_str+".nc")

    #FILE CONCATENATING#
    ds_left = xr.concat([ds1.sel(latitude = slice(10,30-1/50)),
                    ds2.sel(latitude = slice(30,40-1/50)),
                    ds3],dim="latitude")
    ds_left = ds_left.drop_duplicates(dim="latitude")
    ds_left = ds_left.drop_duplicates(dim="longitude")
    ds4 = ds4.drop_duplicates(dim="latitude")

    ds = xr.concat([ds_left.sel(longitude = slice(180,240-1/50)),ds4],dim="longitude")

    #FILE SAVING#

    ds.to_netcdf("/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_"+date_str+".nc")

    del ds1, ds2, ds3, ds4, ds




