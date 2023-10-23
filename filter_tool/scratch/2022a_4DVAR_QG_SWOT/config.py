#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

name_experiment = '2022a_4DVAR_QG_SWOT'

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime,timedelta
 
#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = name_experiment, # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = name_experiment, # name of output files

    path_save = f'../outputs/{name_experiment}', # path of output files

    tmp_DA_path = f"../scratch/{name_experiment}", # temporary data assimilation directory path,

    init_date = datetime(2012,2,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,4,1,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=1),  

    saveoutput_time_step = timedelta(hours=12),  # time step at which the states are saved 

    flag_plot = 1,

    write_obs = True, # the observation files are very low to process, so we decide to save the extracted informations in *path_obs* to read it for several experiments

    path_obs = f'../obs/{name_experiment}'

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_CAR',

    lon_min = 230.,                                        # domain min longitude

    lon_max = 239.,                                        # domain max longitude

    lat_min = 30.,                                         # domain min latitude

    lat_max = 39.,                                         # domain max latitude

    dx = 25.,                                              # grid spacing in km

    name_init_mask = "/home/bellemva/CCS/data/dc_nr/ssh_raw_2012-01-01.nc",

    name_var_mask = {'lon':'lon','lat':'lat','var':'ssh'}

)
#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD1'

myMOD1 = dict(

    super = 'MOD_QG1L_JAX',

    name_var = {'SSH':'ssh'},

    dtmodel = 1200, # model timestep

    time_scheme = 'Euler',

    c0 = 2.7,
    
)


#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = 'myBC'

myBC = dict(

    super = 'BC_EXT',

    file = '/home/bellemva/CCS/outputs/2022a_4DVAR_ID/2022a_4DVAR_ID_*.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = 'time',

    name_var = {'ssh'},

    name_mod_var = {'ssh'},

    dist_sponge = None  # distance (in km) for which boundary fields are spatially spread close to the borders (useful for MOD_QG1L_NP model)

)

#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = 'myOBSOP'

myOBSOP = dict(

    super = 'OBSOP_INTERP',

    path_save = None, # Directory where to save observational operator

    compute_op = False, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_coast = False,

    mask_borders = False,

)

#################################################################################################################################
# Reduced basis parameters
#################################################################################################################################

NAME_BASIS = 'myBASIS1'

myBASIS1 = dict(

    super = 'BASIS_BM',

    flux = False, # True if using 4Dvar with Identity model 

    save_wave_basis = True, # save the basis matrix in tmp_DA_path. If False, the matrix is stored in line

    wavelet_init = True, # Estimate the initial state 

    name_mod_var = 'ssh',

    facns = 1., #factor for wavelet spacing= space

    facnlt = 2., #factor for wavelet spacing= time

    npsp= 3.5, # Defines the wavelet shape

    facpsp= 1.5, # factor to fix df between wavelets

    lmin= 80., # minimal wavelength (in km)

    lmax= 960., # maximal wavelength (in km)

    lmeso = 300, # Largest mesoscale wavelenght 

    tmeso = 10, # Largest mesoscale time of decorrelation 

    sloptdec = -1., # Slope such as tdec = lambda^slope where lamda is the wavelength

    factdec = .5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 0., # minimum time of decorrelation 

    tdecmax = 20., # maximum time of decorrelation 

    facQ= 1, # factor to be multiplied to the estimated Q

    Qmax = .03 , # Maximim Q, such as lambda>lmax => Q=Qmax where lamda is the wavelength

    slopQ = -2 # Slope such as Q = lambda^slope where lamda is the wavelength

)


#################################################################################################################################
# Analysis parameters
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_4DVAR',

    compute_test = False, # TLM, ADJ & GRAD tests

    gtol = 1e-3, # Gradient norm must be less than gtol before successful termination.

    maxiter = 100, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = False, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=6), #  timesteps separating two consecutive analysis 

    sigma_R = 1e-2, # Observational standard deviation

    prec = True, # preconditoning,
 
)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['SWOT']


SWOT = dict(

    super = 'OBS_SSH_SWATH',

    path = '/home/bellemva/CCS/data/dc_obs_swot/2022a_SSH_mapping_CalXover_swot.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',

    name_var = {'SSH':'ssh_model'},

)

'''
ALG = dict(

    super = 'OBS_SSH_NADIR',

    path = '/home/bellemva/CCS/data/dc_obs_nadirs/alg/2022a_SSH_mapping_CalXover_alg.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

C2 = dict(

    super = 'OBS_SSH_NADIR',

    path = '/home/bellemva/CCS/data/dc_obs_nadirs/c2/2022a_SSH_mapping_CalXover_c2.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

J3 = dict(

    super = 'OBS_SSH_NADIR',

    path = '/home/bellemva/CCS/data/dc_obs_nadirs/j3/2022a_SSH_mapping_CalXover_j3.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)

S3A = dict(

    super = 'OBS_SSH_NADIR',

    path = '/home/bellemva/CCS/data/dc_obs_nadirs/s3a/2022a_SSH_mapping_CalXover_s3a.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_var = {'SSH':'ssh'},

)
'''

#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = 'myDIAG'

myDIAG = dict(

    super = 'DIAG_OSSE',

    name_ref = '/home/bellemva/CCS/data/dc_ref_eval/2022a_SSH_mapping_CalXover_eval*.nc',

    time_min = datetime(2012,2,15,0),

    time_max = datetime(2012,3,15,0),

    lon_min = 230,

    lon_max = 239,

    lat_min = 30,

    lat_max = 39,

    name_ref_time = 'time',

    name_ref_lon = 'lon',

    name_ref_lat = 'lat',

    name_ref_var = 'ssh',

    options_ref = {'parallel':True},

    name_exp_var = 'ssh'

)
