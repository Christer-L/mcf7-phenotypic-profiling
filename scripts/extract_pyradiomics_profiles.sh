#!/bin/bash

#SBATCH -o slurm/slurm.%x.%N.%j.out  # STDOUT file with the Job name, the Node name and the Job ID
#SBATCH -e slurm/slurm.%x.%N.%j.err  # STDERR file with the Job name, the Node name and the Job ID
#SBATCH --job-name=feature_extraction
#SBATCH --time=10:00:00
#SBATCH --ntasks=12
#SBATCH --mem=96G

conda run -n pyradiomics python ../src/extract_pyradiomics_profiles.py