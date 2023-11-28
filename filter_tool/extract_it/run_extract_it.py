import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.fftpack as fp
#from scipy.interpolate import RegularGridInterpolator, griddata
from joblib import Parallel
from joblib import delayed as jb_delayed
from pyinterp import fill, Axis, TemporalAxis, Grid3D, Grid2D
from math import *
import glob
import xrft
from datetime import datetime

######################
#### LOADING DATA ####
######################

list_files = glob.glob("/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_201205*.nc")+\
             glob.glob("/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_201206*.nc")+\
             glob.glob("/bettik/bellemva/MITgcm/MITgcm_filtered_final/MITgcm_filt_201207*.nc")

n_try = 0
while n_try<10:
    try : 
        ds = xr.open_mfdataset(list_files,combine='nested',concat_dim='time',parallel=True)
        break
    except : 
        print("Opening netcdf file failed, trying again...")
    n_try+=1

print(datetime.now(),ds)

ds = ds.load()

print("load ok")

###################
#### FUNCTIONS ####
###################

def extract_it(array_ssh,wint,H):
    nt = int(wint.size/3)
    array_ssh=array_ssh.values
    ssh_extended = np.concatenate((np.flip(array_ssh),
                                   array_ssh,
                                   np.flip(array_ssh)))
    ssh_win = wint * ssh_extended 
    ssh_f_t = fp.fft(ssh_win)
    ssh_f_filtered =  H * ssh_f_t
    ssh_filtered = np.real(fp.ifft(ssh_f_filtered))[nt:2*nt]
    del array_ssh
    return ssh_filtered

def gaspari_cohn(array,distance,center):
    """
    NAME 
        bfn_gaspari_cohn

    DESCRIPTION 
        Gaspari-Cohn function. @vbellemin.
        
        Args: 
            array : array of value whose the Gaspari-Cohn function will be applied
            center : centered value of the function 
            distance : Distance above which the return values are zeros


        Returns:  smoothed values 
            
    """ 
    if type(array) is float or type(array) is int:
        array = np.array([array])
    else:
        array = array
    if distance<=0:
        return np.zeros_like(array)
    else:
        array = 2*np.abs(array-center*np.ones_like(array))/distance
        gp = np.zeros_like(array)
        i= np.where(array<=1.)[0]
        gp[i]=-0.25*array[i]**5+0.5*array[i]**4+0.625*array[i]**3-5./3.*array[i]**2+1.
        i =np.where((array>1.)*(array<=2.))[0]
        gp[i] = 1./12.*array[i]**5-0.5*array[i]**4+0.625*array[i]**3+5./3.*array[i]**2-5.*array[i]+4.-2./3./array[i]
        #if type(r) is float:
        #    gp = gp[0]
    return gp

def create_cartesian_grid (latitude,longitude,dx):
    """ 
    Creates a cartesian grid (regular in distance, kilometers) from a geodesic latitude, longitude grid. 
    The new grid is expressed in latitude, longitude coordinates.

    Parameters
    ----------
    longitude : numpy ndarray 
        Vector of longitude for geodesic input grid. 
    latitude : numpy ndarray 
        Vector of latitude for geodesic input grid. 
    dx : float 
        Grid spacing in kilometers. 

    Returns
    -------
    ENSLAT2D : 
        2-D numpy ndarray of the latitudes of the points of the cartesian grid 
    ENSLON2D : 
        2-D numpy ndarray of the longitudes of the points of the cartesian grid 
    """
    km2deg = 1/111

    # ENSEMBLE OF LATITUDES # 
    ENSLAT = np.arange(latitude[0],latitude[-1]+dx*km2deg,dx*km2deg)
    range_lon = longitude[-1]-longitude[0]

    if longitude.size%2 == 0 : 
        nstep_lon = floor(range_lon/(dx*km2deg))+2
    else : 
        nstep_lon = ceil(range_lon/(dx*km2deg))+2
    ENSLAT2D = np.repeat(np.expand_dims(ENSLAT,axis=1),axis=1,repeats=nstep_lon)

    # ENSEMBLE OF LATITUDES # 
    mid_lon = (longitude[-1]+longitude[0])/2
    ENSLON2D=np.zeros_like(ENSLAT2D)

    for i in range(len(ENSLAT)):
        d_lon = dx*km2deg*(np.cos(np.pi*ENSLAT[0]/180)/np.cos(np.pi*ENSLAT[i]/180))
        d_lon_range = np.array([i*d_lon for i in range (1,int(nstep_lon/2)+1)])
        lon_left = np.flip(mid_lon-d_lon_range)
        lon_right = mid_lon+d_lon_range
        ENSLON2D[i,:]=np.concatenate((lon_left,lon_right))

    return ENSLAT2D, ENSLON2D, ENSLAT2D.shape[0], ENSLAT2D.shape[1]

def interpolate_ssh_it(ssh_it):

    x_axis = Axis(ssh_it.longitude.values,is_circle=True)
    y_axis = Axis(ssh_it.latitude.values,is_circle=True)
    t_axis = TemporalAxis(ssh_it.time.values)

    grid = Grid3D(y_axis, x_axis, t_axis, ssh_it.values.transpose(1,2,0))
    has_converged, filled = fill.gauss_seidel(grid,num_threads=4)

    ssh_it_filled = ssh_it.copy(deep=True,data=filled.transpose(2,0,1)).chunk({'time':1})

    dx = 2 # in kilometers, spacing of the grid 

    ENSLAT2D, ENSLON2D, i_lat, i_lon = create_cartesian_grid(ssh_it_filled.latitude.values,
                                                            ssh_it_filled.longitude.values,
                                                            dx)

    array_cart_ssh = ssh_it_filled.interp(latitude=('z',ENSLAT2D.flatten()),
                                        longitude=('z',ENSLON2D.flatten()),
                                        ).values

    # INTERPOLATION OF NaNs # 
    x_axis = Axis(np.arange(i_lon))
    y_axis = Axis(np.arange(i_lat))
    t_axis = TemporalAxis(ssh_it.time.values)

    grid = Grid3D(y_axis, x_axis, t_axis, array_cart_ssh.reshape((24,i_lat,i_lon)).transpose(1,2,0))
    has_converged, filled = fill.gauss_seidel(grid,num_threads=4)


    # CREATION OF DataArray #
    cart_ssh_it = xr.DataArray(data=filled.transpose(2,0,1),
                            dims=["time","y","x"],
                            coords = dict(
                                time = ssh_it_filled.time.values,
                                #y=(["y"],np.arange(i_lat)),
                                #x=(["x"],np.arange(i_lon))
                                y=np.array([i*dx for i in range (i_lat)]),
                                x=np.array([i*dx for i in range (i_lon)])
                            )).chunk({'time':1})
    
    return cart_ssh_it


print(datetime.now(),"functions computed")

####################
#### PROCESSING ####
####################

array_time = ds.ssh_igw.time.values
nt = array_time.size

# PARAMETERS # 
wint = np.ones(3*nt)
gaspari = gaspari_cohn(np.arange(0,2*nt,1),nt,nt)
wint[:nt]=gaspari[:nt]
wint[2*nt:]=gaspari[nt:]

dt = 3600 # seconds

w = fp.fftfreq(3*nt,dt)# seconds^-1
nw = w.size

w1 = 1/15/3600
w2 = 1/9/3600
H = (np.abs(w)>w1) & (np.abs(w)<w2)
w_filtered = H*w

idx_ocean = np.where(np.invert(np.isnan(ds.ssh_igw[0,:,:].values))) # indexes of ocean pixels 

print(datetime.now(),"starting parallel computing")

ssh_it_flat = np.array(Parallel(n_jobs=34,backend='multiprocessing')(jb_delayed(extract_it)(ds.ssh_igw[:,i,j],wint,H) for i,j in zip(idx_ocean[0][:34],idx_ocean[1][:34])))

print(datetime.now(),ssh_it_flat)




##### TRYING PARALLELIZATION #####
"""
import time 
from datetime import datetime 
time0 = datetime.now()

def func(i):
    time.sleep(10)
    return i 

Parallel(n_jobs=64,backend='multiprocessing')(jb_delayed(func)(i) for i in range(64))

print(datetime.now()-time0)
"""
##################################
