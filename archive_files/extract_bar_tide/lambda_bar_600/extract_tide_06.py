import os

import xarray as xr
import numpy as np

import scipy.fftpack as fp

import pyinterp.backends.xarray
import pyinterp.fill

import glob

import gc

from datetime import datetime


date_array = np.arange(np.datetime64("2012-06-01"),np.datetime64("2012-07-01"))


###########################################################################
################################ FUNCTIONS ################################
###########################################################################

def lonlat2dxdy(lon,lat):
    dlon = np.gradient(lon)
    dlat = np.gradient(lat)
    dx = np.sqrt((dlon[1]*111000*np.cos(np.deg2rad(lat)))**2
                 + (dlat[1]*111000)**2)
    dy = np.sqrt((dlon[0]*111000*np.cos(np.deg2rad(lat)))**2
                 + (dlat[0]*111000)**2)
    dx[0,:] = dx[1,:]
    dx[-1,: ]= dx[-2,:] 
    dx[:,0] = dx[:,1]
    dx[:,-1] = dx[:,-2]
    dy[0,:] = dy[1,:]
    dy[-1,:] = dy[-2,:] 
    dy[:,0] = dy[:,1]
    dy[:,-1] = dy[:,-2]
    return dx,dy

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


def extend(ssh,nx,ny):
    ssh_extended = np.empty((3*ny,3*nx))
    ssh_extended[ny:2*ny,nx:2*nx] = +ssh
    ssh_extended[0:ny,nx:2*nx] = +ssh[::-1,:]
    ssh_extended[2*ny:3*ny,nx:2*nx] = +ssh[::-1,:]
    ssh_extended[:,0:nx] = ssh_extended[:,nx:2*nx][:,::-1]
    ssh_extended[:,2*nx:3*nx] = ssh_extended[:,nx:2*nx][:,::-1]
    return ssh_extended

def highpass(_lambda,nx,ny,wavnum2D):
    _highpass = np.zeros((3*ny,3*nx))
    for i in range(3*ny):
        for j in range(3*nx):
            if wavnum2D[i,j]>1/_lambda:
                _highpass[i,j] = 1
    return _highpass

def lowpass(_lambda,nx,ny,wavnum2D) : 
    _lowpass = np.zeros((3*ny,3*nx))
    for i in range (3*ny):
        for j in range(3*nx):
            if wavnum2D[i,j]<1/_lambda:
                _lowpass[i,j] = 1
    return _lowpass 

def apply_filter(ssh_freq,H):
    ssh_freq_filtered = H * ssh_freq
    ssh_filtered = np.real(fp.ifft2(ssh_freq_filtered))
    return ssh_filtered

def interpolate_nans(_ds):
    has_converged, filled = pyinterp.fill.gauss_seidel(_ds)
    if has_converged :
        return filled.T
    else : 
        "Pyinterp hasn't converged"
        raise Exception



def save_bar_tide_hour(day_str,time_index,ds) :

    longitude = ds.longitude.values
    latitude = ds.latitude.values

    x_axis = pyinterp.Axis(longitude)
    y_axis = pyinterp.Axis(latitude)

    time = ds.time.values
    ny = latitude.size
    nx = longitude.size

    ssh0 = ds.values
    mask = np.isnan(ssh0)

    lon2d,lat2d = np.meshgrid(longitude,latitude)
    dx,dy = lonlat2dxdy(lon2d,lat2d)
    dx0 = dx.mean()
    dy0 = dy.mean()

    kx = np.fft.fftfreq(3*nx,dx0*1e-3) # km
    ky = np.fft.fftfreq(3*ny,dy0*1e-3) # km
    k, l = np.meshgrid(kx,ky)
    wavnum2D = np.sqrt(k**2 + l**2)

    winy = np.ones(3*ny)
    winy[:ny] = gaspari_cohn(np.arange(0,ny,1),ny)[::-1]
    winy[2*ny:] = gaspari_cohn(np.arange(0,ny),ny)

    winx = np.ones(3*nx)
    winx[:nx] = gaspari_cohn(np.arange(0,nx,1),nx)[::-1]
    winx[2*nx:] = gaspari_cohn(np.arange(0,nx),nx)

    window = winy[:,np.newaxis] * winx[np.newaxis,:]

    lambda_bar = 600
    lowpass_BAR = lowpass(lambda_bar,nx,ny,wavnum2D) 

    # Remove nans
    ssh_noNans = interpolate_nans(pyinterp.Grid2D(x_axis, y_axis, ssh0.T))

    # Extend
    ssh_extended = extend(ssh_noNans,nx,ny)        

    # Windowing
    ssh_win = ssh_extended * window

    # FFT
    ssh_freq = fp.fft2(ssh_win)

    # Filter
    ssh_bar = apply_filter(ssh_freq, lowpass_BAR)[ny:2*ny,nx:2*nx]

    # Mask
    ssh_bar[mask] = np.nan

    ssh_bar = np.expand_dims(ssh_bar,axis=0)

    dsout = xr.Dataset({'ssh_bar':(('time','latitude','longitude'),xr.DataArray(ssh_bar, dims=['time','lat','lon']).data),}, 
                    coords={'time':('time',np.array([time])),'latitude':('latitude',latitude),'longitude':('longitude',longitude)} 
                    )

    dsout=dsout.interp({'longitude':np.round(np.arange(180.0,245.0+1/50,1/48),decimals=10),'latitude':np.round(np.arange(10.0,45.0+1/50,1/48),decimals=10)},method='linear')

    dsout.to_netcdf("/bettik/bellemva/MITgcm/MITgcm_bar/MITgcm_bar_"+day_str+"T"+str(time_index)+".nc")


    del dsout, ssh_bar 


def compute_bar_tide_day(date): 

    ds = xr.open_dataset("/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_"+date.astype('str').replace("-","")+".nc")
    mask = np.load("/bettik/bellemva/MITgcm/mask/mask_MITgcm.npy")
    ds = ds.where(np.invert(mask))

    for time_index in range (0,24):

        rolling = ds.ssh_hf[time_index].rolling({'longitude':5,'latitude':5},center=True,min_periods=1).mean()
        ds_deg = rolling.interp({'longitude':np.round(np.arange(180.0,245.0+1/13,1/12),decimals=10),'latitude':np.round(np.arange(10.0,45.0+1/13,1/12),decimals=10)})
        save_bar_tide_hour(day_str=date.astype('str').replace("-",""),time_index=time_index,ds = ds_deg)

    del ds, ds_deg
    gc.collect()



###########################################################################
################################   CALL   ################################
###########################################################################


time1 = datetime.now()

for date in date_array : 

    compute_bar_tide_day(date)

print("done :",datetime.now()-time1)