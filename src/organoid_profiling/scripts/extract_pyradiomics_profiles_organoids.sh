#!/bin/bash

#SBATCH -o slurm.%x.%N.%j.out  # STDOUT file with the Job name, the Node name and the Job ID
#SBATCH -e slurm.%x.%N.%j.err  # STDERR file with the Job name, the Node name and the Job ID
#SBATCH --job-name=feature_extraction
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=14G

cd ..
cd ..
cd ..

conda run -n pyradiomics python -m src.organoid_profiling.extract_pyradiomics_profiles_organoids