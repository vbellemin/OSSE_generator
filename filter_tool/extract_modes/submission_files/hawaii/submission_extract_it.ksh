#!/bin/bash

#OAR -n run_hawaii
#OAR -l nodes=1/core=24,walltime=07:00:00
#OAR --stdout run_hawaii.out
#OAR --stderr run_hawaii.err
#OAR --project data-ocean

source /home/bellemva/miniconda3/etc/profile.d/conda.sh
conda activate pangeo-forge 
cd /home/bellemva/OSSE_generator/filter_tool/extract_modes/submission_files/hawaii
python run.py