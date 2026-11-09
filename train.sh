GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CONFIG=${1:-local_configs.NYUDepthv2.DFormerPP_B}
export DATASET="DeStijl_occ"

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1

for augmentatie in --augmentatie; do
    for seed in $(seq 0 4); do
        for ds_size in 1 50 10 5 1; do
            for mask in --mask --no-mask; do
            echo ">>> mask=$mask seed=$seed ds_size=$ds_size"
            PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
            torchrun \
                --nnodes=$NNODES \
                --node_rank=$NODE_RANK \
                --master_addr=$MASTER_ADDR \
                --nproc_per_node=$GPUS \
                --master_port=$PORT \
                utils/train.py \
                --config=$CONFIG --gpus=$GPUS \
                --no-sliding \
                --no-compile \
                --syncbn \
                --no-mst \
                --compile_mode="default" \
                --no-amp \
                --val_amp \
                --ds_size=$ds_size \
                $mask \
                $augmentatie \
                --use_seed \
                --seed=$seed
            done
        done
    done
done


# Usage:
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_B
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_S
#   bash train.sh local_configs.NYUDepthv2.DFormerPP_T
#   bash train.sh local_configs.NYUDepthv2.DFormer_Base
#   bash train.sh local_configs.NYUDepthv2.DFormerv2_B
#   bash train.sh local_configs.SUNRGBD.DFormerPP_B
