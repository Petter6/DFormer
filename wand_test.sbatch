export DATASET="FashionOcc_200"
export DS_SIZE=200
export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}

wandb agent --count 1 reijaldinho-tu-delft/FashionMnist-occ/6a9ha69h