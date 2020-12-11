#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=cz2064@nyu.edu

module purge
module load anaconda3/5.3.1
source activate /scratch/cz2064/envs/dl4med

srun python -u 1_preprocess_seg.py
