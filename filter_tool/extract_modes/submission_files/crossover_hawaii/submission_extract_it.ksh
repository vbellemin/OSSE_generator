#!/bin/bash

#OAR -n run_crossover_hawaii
#OAR -l nodes=1/core=24,walltime=02:00:00
#OAR --stdout run_crossover_hawaii.out
#OAR --stderr run_crossover_hawaii.err
#OAR --project data-ocean

source /home/bellemva/miniconda3/etc/profile.d/conda.sh
conda activate pangeo-forge 
cd /home/bellemva/OSSE_generator/filter_tool/extract_modes/submission_files/crossover_hawaii
python run.py