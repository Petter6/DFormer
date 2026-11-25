#!/bin/bash
#SBATCH --job-name=occ-nomask
#SBATCH --account=daic-eemcs-insy-prb
#SBATCH --partition=daic-eemcs-insy-prb,daic-eemcs-insy,daic-all,all
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -uo pipefail

SETTINGS=(
  "FashionMetOcc_100 --no-mask 20 1e-4 --augmentaties"
  "FashionMetOcc_200 --no-mask 20 1e-4 --augmentaties"
  "FashionMetOcc_500 --no-mask 20 1e-4 --augmentaties"
  "FashionZonderOcc_100 --no-mask 20 1e-4 --augmentaties"
  "FashionZonderOcc_200 --no-mask 20 1e-4 --augmentaties"
  "FashionZonderOcc_500 --no-mask 20 1e-4 --augmentaties"
)

# sbatch: array-index; interactief: eerste argument (default 0)
TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
read -r DATASET MASK BS LR AUG <<< "${SETTINGS[$TASK_ID]:-}"
[ -z "${DATASET:-}" ] && { echo "Geen setting voor index $TASK_ID"; exit 1; }
echo "${DATASET} ${MASK} ${BS} ${LR} ${AUG}"

export DATASET
export DS_SIZE=${DATASET##*_}
export BS
export LR

MY=/tudelft.net/staff-homes-linux/r/pjwreijalt
export WANDB_DIR=$MY/DFormer/wandb
mkdir -p "$WANDB_DIR"

# sbatch: map waar je sbatch aanriep; interactief: huidige map
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO":${PYTHONPATH:-}

CONFIG="local_configs.DeStijl.DFormerv2_S"
GPUS=1
SEED=0

apptainer exec --nv --home "$MY" ./container.sif \
  torchrun --standalone --nproc_per_node=$GPUS \
    utils/train.py \
    --config=$CONFIG --gpus=$GPUS \
    --no-sliding --no-compile --syncbn --no-mst \
    --compile_mode="default" --no-amp --val_amp \
    $MASK $AUG \
    --use_seed --seed=$SEED