import os 
import xarray as xr
import numpy as np 
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import scipy.fftpack as fp
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime 

from dask import delayed,compute

import glob 

from dask.distributed import Client, LocalCluster

from dask import config as cfg 

#cfg.set({'distributed.scheduler.worker-ttl': None})



path05 = "/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_201205*.nc"
path06 = "/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_201206*.nc"
path07 = "/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_201207*.nc"

path = np.sort(np.concatenate([glob.glob(path05),glob.glob(path06),glob.glob(path07)]))

ds_ssh = xr.open_mfdataset(path, combine="nested",concat_dim="time")

ds_ssh = ds_ssh.sel(longitude=slice(215,245),latitude=slice(15,45))
#ds_ssh = ds_ssh.sel(time=slice(np.datetime64("2012-05-01"),np.datetime64("2012-05-03")))

longitude = ds_ssh.longitude.values
latitude = ds_ssh.latitude.values
time= ds_ssh.time.values
ny = latitude.size
nx = longitude.size
nt = time.size

def compute_band_pass(ssh,t1,t2): 

    """
    This function computes a band-stop filter over a the Sea Surface Height Timesery ssh. 
    Timestep is set in the variable dt. 
    """

    nt = ssh.size

    if t1 > t2 :
        print("t1 should be the smallest cutoff frequency")
        raise ValueError

    dt = 3600 # seconds
 
    w = fp.fftfreq(3*nt,dt)# seconds^-1

    w1 = 1/15/3600
    w2 = 1/9/3600
    H = (np.abs(w)>w1) & (np.abs(w)<w2)

    wint = np.ones(3*nt)
    wint[:nt] = gaspari_cohn(np.arange(0,nt,1),nt)[::-1]
    wint[2*nt:] = gaspari_cohn(np.arange(0,nt),nt)
    
    ssh_extended = np.zeros((3*nt,))
    ssh_extended[nt:2*nt] = ssh
    ssh_extended[:nt] = ssh[::-1]
    ssh_extended[2*nt:] = ssh[::-1]
    ssh_win = wint * ssh_extended 
    ssh_f_t = fp.fft(ssh_win)
    ssh_f_filtered =  H * ssh_f_t
    ssh_filtered = np.real(fp.ifft(ssh_f_filtered))[nt:2*nt]
    return ssh_filtered


def gaspari_cohn(r,c):

    if type(r) is float or type(r) is int:
        ra = np.array([r])
    else:
        ra = r
    if c<=0:
        return np.zeros_like(ra)
    else:
        ra = 2*np.abs(ra)/c
        gp = np.zeros_like(ra)
        i= np.where(ra<=1.)[0]
        gp[i]=-0.25*ra[i]**5+0.5*ra[i]**4+0.625*ra[i]**3-5./3.*ra[i]**2+1.
        i =np.where((ra>1.)*(ra<=2.))[0]
        gp[i] = 1./12.*ra[i]**5-0.5*ra[i]**4+0.625*ra[i]**3+5./3.*ra[i]**2-5.*ra[i]+4.-2./3./ra[i]
        if type(r) is float:
            gp = gp[0]
    return gp

def extract_it (data_array, i_lon, i_lat) : 
    ssh_filt = compute_band_pass(data_array[:,i_lat,i_lon].values,9,15)
    np.save("/home/bellemva/CCS/data/filter_MITgcm/extract_IT_modes/"+str(i_lon)+"_"+str(i_lat)+".npy",ssh_filt)
    return ssh_filt 

def run():
    ask_workers = 4
    #cluster = LocalCluster(n_workers=ask_workers,dashboard_address=':8686')
    cluster = LocalCluster()
    c = Client(cluster)

    delayed_results = []
    for i_lon in range(3):
        for i_lat in range (3) : 
            res = delayed(extract_it)(ds_ssh.ssh_igw,i_lon,i_lat)
            delayed_results.append(res)
    results = compute(*delayed_results, scheduler="processes")



if __name__ == '__main__':
    time1 = datetime.now()
    run()
    print(datetime.now()-time1)

