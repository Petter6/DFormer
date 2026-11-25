import wandb

# wandb-sweep setup
# dit aanpassen op de apptainer enz.
sweep_config = {"method": "random"}

metric = {"name": "rm5", "goal": "maximize"}

sweep_config["metric"] = metric

parameters_dict = {
    "lr": {"distribution": "log_uniform_values", "min": 0.0000001, "max": 0.01},
    "batch_size": {"values": [8, 16, 32, 64, 128]},
    "weight_decay": {"distribution": "log_uniform_values", "min": 0.001, "max": 0.1},
    "bn_momentum": {"min": 0.01, "max": 0.2},
    "drop_path_rate": {"min": 0.01, "max": 0.3},
}

sweep_config["parameters"] = parameters_dict

sweep_config["program"] = "utils/train.py"
sweep_config["command"] = [
    "apptainer", "exec", "--nv", "./container.sif",
    "torchrun", "--standalone", "--nproc_per_node=1",
    "${program}",
    "--config=local_configs.DeStijl.DFormerv2_S", "--gpus=1",
    "--no-sliding", "--no-compile", "--syncbn", "--no-mst",
    "--no-amp", "--val_amp", "--mask", "--augmentaties",
    "--use_seed", "--seed=0",
]

sweep_id = wandb.sweep(sweep_config, project="FashionMnist-occ")