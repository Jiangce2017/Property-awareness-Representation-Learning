#!/bin/bash
#SBATCH --job-name=property_vae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jc14407@nyu.edu
#SBATCH --output=slurm_%j.out


#SBATCH --mail-type=END
#SBATCH --mail-user=jc14407@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load anaconda3/2024.02
source $HOME/.bashrc
conda activate /scratch/jc14407/.conda/env/torch
export MPLCONFIGDIR=/scratch/jc14407/.config/matplotlib

python train_epi.py



