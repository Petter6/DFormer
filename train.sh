GPUS=1
CONFIG=${1:-local_configs.NYUDepthv2.DFormerPP_B}
export DATASET="FashionOcc_100"
export DS_SIZE=100

SEED=1
# implementeren lr (3e-5, 1e-4, 3e-4), batch_size (10, 20, 50), augmentatie/geen_augmentatie

for augmentatie in --augmentatie --no-augmentatie ; do
    
    for mask in --no-mask --no-mask; do
        echo ">>> mask=$mask seed=$seed ds_size=$ds_size"
        PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
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
            --lr 3e-5 \
            --batch_size 25 \
            $mask \
            $augmentatie \
            --use_seed \
            --seed=$SEED
    done
done


# Usage:
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_B
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_S
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_T
#   bash train.sh local_configs.NYUDepthv2.DFormer_Base
#   bash train.sh local_configs.NYUDepthv2.DFormerv2_B
#   bash train.sh local_configs.SUNRGBD.DFormerPP_B
