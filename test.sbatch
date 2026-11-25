set -uo pipefail

# Full model list (order doesn’t matter)
SETTINGS=(
  "FashionApp_100 --mask 20 1e-4 --augmentatie"
)

TASK_ID=0
# lees de string in, sla op in variabelen 
read -r DATASET MASK BS LR AUG <<< "${SETTINGS[$TASK_ID]}"

echo "${DATASET} ${MASK} ${BS} ${LR} ${AUG}"

export DATASET
export DS_SIZE=${DATASET##*_}   # haalt 100/200/500 uit de naam
export BS
export LR

CONFIG="local_configs.DeStijl.DFormerv2_S"
GPUS=1
SEED=2

OUT="results/test/${DATASET}_${MASK#--}_bs${BS}_lr${LR}_${AUG#--}.csv"
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
