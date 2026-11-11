#!/bin/bash
#SBATCH --job-name=flow-validate
#SBATCH --partition=prb,insy,general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -uo pipefail

# Full model list (order doesn’t matter)
SETTINGS=("MNISTSeg_100 --mask 20 3e-4 --augmentatie")
read -r DATASET MASK BS LR AUG <<< "${SETTINGS[0]}"
SEED=$SLURM_ARRAY_TASK_ID

echo "${DATASET} ${MASK} ${BS} ${LR} ${AUG}"

export DATASET
export DS_SIZE=${DATASET##*_}   # haalt 100/200/500 uit de naam
export BS
export LR

CONFIG="local_configs.DeStijl.DFormerv2_S"
GPUS=1

OUT="results/${SLURM_ARRAY_JOB_ID}/${DATASET}_${MASK#--}_bs${BS}_lr${LR}_${AUG#--}_seed${SEED}.csv"
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
    --out_file=$OUT
