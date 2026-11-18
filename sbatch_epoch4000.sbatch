#!/bin/bash
#SBATCH --job-name=epoch4000
#SBATCH --partition=prb,insy,general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -uo pipefail

# Full model list (order doesn’t matter)
SETTINGS=("FashionOcc_500 --mask 20 1e-4 --augmentatie")
read -r DATASET MASK BS LR AUG <<< "${SETTINGS[0]}"
SEED=0

echo "${DATASET} ${MASK} ${BS} ${LR} ${AUG}"

export DATASET
export DS_SIZE=${DATASET##*_}   # haalt 100/200/500 uit de naam
export BS
export LR

CONFIG="local_configs.DeStijl.DFormerv2_S"
GPUS=1

OUT="results/${SLURM_JOB_ID}/${DATASET}_${MASK#--}_bs${BS}_lr${LR}_${AUG#--}_seed${SEED}.csv"
mkdir -p "$(dirname "$OUT")"

export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}

apptainer exec --nv \
  ./container.sif \
  torchrun \
    --standalone \
    --nproc_per_node=$GPUS \
    utils/train.py \
    --config=$CONFIG --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --no-mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    $MASK \
    $AUG \
    --use_seed \
    --seed=$SEED \
    --out_file=$OUT \
    --epochs 4000
