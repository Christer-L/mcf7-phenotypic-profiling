#!/bin/bash

#SBATCH -o slurm.%x.%N.%j.out  # STDOUT file with the Job name, the Node name and the Job ID
#SBATCH -e slurm.%x.%N.%j.err  # STDERR file with the Job name, the Node name and the Job ID
#SBATCH --job-name=object_extraction
#SBATCH --time=10:00:00
#SBATCH --ntasks=4
#SBATCH --mem=80G

cd ..

conda run -n cells2circles python -m extract_organoids.py