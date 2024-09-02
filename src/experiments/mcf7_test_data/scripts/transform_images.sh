#!/bin/bash

#SBATCH -o slurm.%x.%N.%j.out  # STDOUT file with the Job name, the Node name and the Job ID
#SBATCH -e slurm.%x.%N.%j.err  # STDERR file with the Job name, the Node name and the Job ID
#SBATCH --job-name=transform_images
#SBATCH --time=10:00:00
#SBATCH --ntasks=32
#SBATCH --mem=190G

cd ../../../..

pwd

conda run -n cells2circles python -m src.experiments.mcf7_test_data.2_transform_images

