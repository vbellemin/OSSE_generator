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

#from dask import delayed,compute

import sys
import gc 

lat0 = 10

print("lat0 = ",lat0,"\n\n")

############# CALCULATING HF AND BM ###############

dir_input = '/bettik/bellemva/MITgcm/MITgcm_dedac'
pattern = 'MITgcm_'

ds_ssh = xr.open_mfdataset(os.path.join(dir_input,pattern+'*.nc'))

#ds_ssh = ds_ssh.chunk({'longitude':5*48,'latitude':5*48,'time':ds_ssh.time.size})

ds_ssh = ds_ssh.sel(latitude=slice(10,20),longitude=slice(220,230))

print("ds_ssh ok \n \n")

longitude = ds_ssh.longitude.values
latitude = ds_ssh.latitude.values
time = ds_ssh.time.values
nt = time.size
ny = latitude.size
nx = longitude.size

# Coriolis period
f = 2*2*np.pi/86164*np.sin(np.mean(np.deg2rad(latitude))) 
T = 2*np.pi/f

dt = 3600 # number of seconds in a hour 
window_len = int(2*T//dt) # length of the gaussian filter window 
time_window = np.arange(-window_len,window_len+1) # array time steps to compute the kernel 
exp_window = np.exp(-np.square(time_window/(T/dt))) # array of kernel values 
ntw = time_window.size

weight = xr.DataArray(exp_window, dims=['window'])
ssh_dedac = ds_ssh.ssh_dedac#.chunk({'longitude':5*48,'latitude':5*48,'time':ds_ssh.time.size})#[:nt-nt%ntw]
ssh_bm = ssh_dedac.rolling(time=ntw, center=True).construct('window').dot(weight)/weight.sum()
#ssh_bm = ssh_bm.chunk({'longitude':5*48,'latitude':5*48,'time':-1})

#ssh_hf = ssh_dedac - ssh_bm
#ssh_hf = ssh_hf.chunk({'longitude':5*48,'latitude':5*48,'time':ssh_dedac.time.size})


ssh_bm = ssh_bm.load()
ssh_hf = ssh_dedac - ssh_bm

dsout = xr.Dataset({'ssh_dedac':(('time','latitude','longitude'),ssh_dedac.data),
                'ssh_bm':(('time','latitude','longitude'),ssh_bm.data),
                'ssh_hf':(('time','latitude','longitude'),ssh_hf.data)
                },
                coords={'time':('time',ssh_dedac.time.values),'latitude':('latitude',ssh_dedac.latitude.values),'longitude':('longitude',ssh_dedac.longitude.values)}
                )

name_file = "MITgcm_filt_test.nc"
dir_output = "/bettik/bellemva/MITgcm/MITgcm_filtered"
dsout.to_netcdf(os.path.join(dir_output,name_file))

#del _ssh_dedac, _ssh_bm 
gc.collect()






print("separation hf bm ok \n \n")

