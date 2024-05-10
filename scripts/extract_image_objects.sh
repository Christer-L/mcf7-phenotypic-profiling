#!/bin/bash

#SBATCH -o slurm/slurm.%x.%N.%j.out  # STDOUT file with the Job name, the Node name and the Job ID
#SBATCH -e slurm/slurm.%x.%N.%j.err  # STDERR file with the Job name, the Node name and the Job ID
#SBATCH --job-name=object_extraction
#SBATCH --time=10:00:00
#SBATCH --ntasks=30
#SBATCH --mem=196G

conda run -n cells2circles python ../src/extract_nuclei.py