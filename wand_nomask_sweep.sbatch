#!/bin/bash
#SBATCH --job-name=mask-sweep
#SBATCH --account=daic-eemcs-insy-prb
#SBATCH --partition=ewi-insy
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --mem=16G
#SBATCH --array=0-20
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -uo pipefail

export DATASET="FashionOcc_500"
export DS_SIZE=500
export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}
export WANDB_DIR=/tudelft.net/staff-homes-linux/r/pjwreijalt/DFormer/wandb
MY=/tudelft.net/staff-homes-linux/r/pjwreijalt

cd "$SLURM_SUBMIT_DIR"
apptainer exec --nv --home $MY ./container.sif wandb agent --count 1 reijaldinho-tu-delft/FashionMnist-occ/qf8up29m