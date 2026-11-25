export DATASET="FashionOcc_500"
export DS_SIZE=500
export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}
export WANDB_DIR=/tudelft.net/staff-homes-linux/r/pjwreijalt/DFormer/wandb
MY=/tudelft.net/staff-homes-linux/r/pjwreijalt

cd "$SLURM_SUBMIT_DIR"
apptainer exec --nv --home $MY ./container.sif wandb agent --count 1 reijaldinho-tu-delft/FashionMnist-occ/q9lmx103