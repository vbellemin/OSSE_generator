#!/bin/bash

#OAR -n extract_it
#OAR -l nodes=1/core=32,walltime=00:00:00
#OAR --stdout extract_it.out
#OAR --stderr extract_it.err
#OAR --project data-ocean

source /home/bellemva/miniconda3/etc/profile.d/conda.sh
conda activate pangeo-forge 
cd /home/bellemva/OSSE_generator/filter_tool/extract_it
python run_extract_it.py