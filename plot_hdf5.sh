#! /bin/bash
#SBATCH --mail-type=ALL
#SBATCH -p compute-64-512
#SBATCH --time=0-01:00
#SBATCH --job-name=atlas
#SBATCH -o atlas-%j.out
#SBATCH -e atlas-%j.err
module add python/anaconda/2023.07/3.11.4
source activate atlas
python plot_hdf5.py

