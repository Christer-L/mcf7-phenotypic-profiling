#!/bin/bash

#SBATCH -o slurm.%x.%N.%j.out  # STDOUT file with the Job name, the Node name and the Job ID
#SBATCH -e slurm.%x.%N.%j.err  # STDERR file with the Job name, the Node name and the Job ID
#SBATCH --job-name=feature_extraction
#SBATCH --time=10:00:00
#SBATCH --ntasks=4
#SBATCH --mem=60G

cd ..
cd ..
cd ..
cd ..

pwd

conda run -n pyradiomics python -m src.experiments.yeast.extract_pyradiomics_profiles

