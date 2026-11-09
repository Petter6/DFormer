import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
import random

train_samples = 100
test_samples = 20
offset = 50
margin = 1  # breedte van de border aan elke kant van de rand
seed = 0
nr_samples = train_samples + test_samples
h_img, w_img = 224, 224
dir_path = "./datasets"
random_loc = True

dataset_name = "DeStijl"

path = Path(dir_path) / dataset_name

np.random.seed(seed)
torch.manual_seed(seed)

for sub in ["Depth", "Label", "RGB"]:
    os.makedirs(f"./{dir_path}/{dataset_name}/{sub}", exist_ok=True)

images = torch.rand((nr_samples, 3, h_img, w_img), dtype=torch.float)

# depths = torch.full((nr_samples, h_img, w_img), fill_value=254, dtype=torch.uint8)
depths = torch.full((nr_samples, h_img, w_img), fill_value=254, dtype=torch.uint8)
labels = torch.ones((nr_samples, h_img, w_img), dtype=torch.uint8)


for idx in range(nr_samples):
    rand_rgb = torch.rand(size=(3, 1, 1))
    images[idx, :, :, :] = rand_rgb

    x_offset = np.random.randint(-offset, offset) if random_loc else 0
    y_offset = np.random.randint(-offset, offset) if random_loc else 0

    low_x, high_x = 94 + x_offset, 130 + x_offset  # high is exclusief
    low_y, high_y = 94 + y_offset, 130 + y_offset

    images[idx, :, low_y:high_y, low_x:high_x] = torch.tensor(
        (np.random.random(), 0, 0)
    ).view(3, 1, 1)

    depths[idx, low_y:high_y, low_x:high_x] = 1

    random_getal = random.random()

    if random_getal > 0.5:
        labels[idx, low_y:high_y, low_x:high_x] = 2
        low_x, high_x = 130 + x_offset, 166 + x_offset  # high is exclusief
        low_y, high_y = 130 + y_offset, 166 + y_offset

        images[idx, :, low_y:high_y, low_x:high_x] = torch.tensor(
            (np.random.random(), 0, 0)
        ).view(3, 1, 1)
        labels[idx, low_y:high_y, low_x:high_x] = 2
    else:
        labels[idx, low_y:high_y, low_x:high_x] = 3
        low_x, high_x = 58 + x_offset, 94 + x_offset  # high is exclusief
        low_y, high_y = 130 + y_offset, 166 + y_offset

        images[idx, :, low_y:high_y, low_x:high_x] = torch.tensor(
            (np.random.random(), 0, 0)
        ).view(3, 1, 1)

        labels[idx, low_y:high_y, low_x:high_x] = 3

    depths[idx, low_y:high_y, low_x:high_x] = 1


palette = [0, 0, 0, 128, 128, 128, 220, 20, 60]

for i in range(nr_samples):
    torchvision.utils.save_image(images[i], path / "RGB" / f"{i}.png")

    Image.fromarray(depths[i].numpy()).save(path / "Depth" / f"{i}.png")

    lbl = Image.fromarray(labels[i].numpy())  # mode 'L', values 1/2
    lbl.save(path / "Label" / f"{i}.png")


def lst_line(i):
    return "\t".join(str(f"{sub}/{i}.png") for sub in ["RGB", "Label"]) + "\n"


with open(path / "train.txt", "w") as f:
    for i in range(train_samples):
        f.write(lst_line(i))

with open(path / "test.txt", "w") as f:
    for i in range(train_samples, nr_samples):
        f.write(lst_line(i))

print(labels.shape)
print(labels.shape)
