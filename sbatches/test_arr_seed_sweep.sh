#!/bin/bash
# Testversie van arr_seed_sweep.sbatch voor een interactieve sessie.
# Gebruik:  bash test_arr_seed_sweep.sh [index]     (index = welke setting; default 0)
set -uo pipefail

# Full model list (order doesn’t matter)
SETTINGS=(
  "FashionOcc_500 --mask 20 1e-4 --augmentatie"
)

TASK_ID=${1:-0}
# lees de string in, sla op in variabelen 
read -r DATASET MASK BS LR AUG <<< "${SETTINGS[$TASK_ID]:-}"
[ -z "${DATASET:-}" ] && { echo "Geen setting voor index $TASK_ID (0..$(( ${#SETTINGS[@]} - 1 )))"; exit 1; }

echo "${DATASET} ${MASK} ${BS} ${LR} ${AUG}"

export DATASET
export DS_SIZE=${DATASET##*_}   # haalt 100/200/500 uit de naam
export BS
export LR

CONFIG="local_configs.DeStijl.DFormerv2_S"
GPUS=1
SEED=${2:-0}   # seed als tweede argument (sweep: 0..4)

OUT="results/test/${DATASET}_${MASK#--}_bs${BS}_lr${LR}_${AUG#--}_seed${SEED}.csv"
mkdir -p "$(dirname "$OUT")"

REPO="$(cd "$(dirname "$0")" && pwd)"     # map waar dit script staat = repo-root
export PYTHONPATH="$REPO":${PYTHONPATH:-}

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
