#!/bin/bash
#SBATCH --job-name=flow-validate
#SBATCH --partition=prb,insy,general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --array=0-18
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

module use /opt/insy/modulefiles
module load cuda/12.4
module load devtoolset/11


# Full model list (order doesn’t matter)
SETTINGS=(
  "FashionOcc_100 --mask 20 1e-4 --augmentatie"
  "FashionOcc_200 --mask 20 1e-4 --augmentatie"
  "FashionOcc_500 --mask 20 1e-4 --augmentatie"
  "MNISTSeg_100 --mask 20 1e-4 --augmentatie"
  "MNISTSeg_200 --mask 20 1e-4 --augmentatie"
  "MNISTSeg_500 --mask 20 1e-4 --augmentatie"
  "FashionOcc_100 --no-mask 20 1e-4 --augmentatie"
  "FashionOcc_200 --no-mask 20 1e-4 --augmentatie"
  "FashionOcc_500 --no-mask 20 1e-4 --augmentatie"
  "MNISTSeg_100 --no-mask 20 1e-4 --augmentatie"
  "MNISTSeg_200 --no-mask 20 1e-4 --augmentatie" 
  "MNISTSeg_500 --no-mask 20 1e-4 --augmentatie" 
  "FashionOcc_100 --mask 10 1e-4 --augmentatie"
  "FashionOcc_100 --mask 50 1e-4 --augmentatie"
  "FashionOcc_100 --mask 20 3e-4 --augmentatie"
  "FashionOcc_100 --mask 20 3e-5 --augmentatie"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}   
# lees de string in, sla op in variabelen 
read -r DATASET MASK BS LR AUG <<< "${SETTINGS[$TASK_ID]}"

echo "${DATASET} ${MASK} ${BS} ${LR} ${AUG}"

export DATASET
export DS_SIZE=${DATASET##*_}   # haalt 100/200/500 uit de naam
export BS
export LR

CONFIG="local_configs.DeStijl.DFormerv2_S"
GPUS=1
SEED=1

OUT="results/${SLURM_ARRAY_JOB_ID}/${DATASET}_${MASK#--}_bs${BS}_lr${LR}_${AUG#--}.csv"
mkdir -p "$(dirname "$OUT")"

PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-} \
torchrun \
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