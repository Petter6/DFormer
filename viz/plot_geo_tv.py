"""Staafdiagram per block van het geo-dieptegewicht (geo_d) en de total variation (tv) voor één epoch.

Gebruik:
    python plot_geo_tv.py results/<job>/..._parameters.csv               # epoch 2000
    python plot_geo_tv.py results/<job>/..._parameters.csv 500            # andere epoch
    python plot_geo_tv.py results/<job>/..._parameters.csv 2000 7a76f262  # specifieke run_id
    python plot_geo_tv.py pad.csv 2000 7a76f262 uit.png                   # eigen uitvoernaam

Verwacht kolommen: run_id, epoch, layer, block, geo_d, tv.
Bestaat de gevraagde epoch niet, dan wordt de laatste volledige epoch gebruikt (met melding).
Twee panelen boven elkaar (geo_d en tv hebben een heel andere schaal), met stagegrenzen.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

GEO_YLIM = (0.85, 1.4)  # vaste y-as voor geo_d
TV_YLIM = (0.0, 0.07)  # vaste y-as voor TV

csv_pad = sys.argv[1] if len(sys.argv) > 1 else "uit_parameters.csv"
gevraagde_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
gekozen_run = sys.argv[3] if len(sys.argv) > 3 else None
uitvoer = sys.argv[4] if len(sys.argv) > 4 else None

df = pd.read_csv(csv_pad)
df = df[df["epoch"].astype(str) != "epoch"]  # eventueel herhaalde headers eruit
df = df.astype({"epoch": int, "layer": int, "block": int, "geo_d": float, "tv": float})

# Run kiezen
if gekozen_run is None:
    gekozen_run = df["run_id"].iloc[-1]
    if df["run_id"].nunique() > 1:
        print(f"Meerdere runs gevonden; ik neem de laatste: {gekozen_run}")
run = df[df["run_id"] == gekozen_run]
if run.empty:
    sys.exit(f"Run '{gekozen_run}' niet gevonden.")

# Epoch kiezen: gevraagde epoch, anders de laatste volledige epoch
n_blocks = run.groupby("epoch").size().max()
volledig = run.groupby("epoch").filter(lambda g: len(g) == n_blocks)
if gevraagde_epoch in set(volledig["epoch"]):
    epoch = gevraagde_epoch
else:
    epoch = volledig["epoch"].max()
    print(
        f"Epoch {gevraagde_epoch} niet (volledig) aanwezig; ik gebruik epoch {epoch}."
    )
data = run[run["epoch"] == epoch].sort_values(["layer", "block"]).reset_index(drop=True)

x = np.arange(len(data))
fig, (ax_geo, ax_tv) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

ax_geo.bar(x, data["geo_d"], color="#DD8452", width=0.75)
ax_geo.axhline(1.0, color="black", linewidth=0.8, linestyle=":", label="init (1.0)")
ax_geo.set_ylabel("geo_d (dieptegewicht)")
ax_geo.set_ylim(*GEO_YLIM)
ax_geo.legend(loc="upper right", fontsize=8)

ax_tv.bar(x, data["tv"], color="#4C72B0", width=0.75)
ax_tv.set_ylabel("TV (fractie verschoven attention)")
ax_tv.set_ylim(*TV_YLIM)

# Gewone getallen op de as (geen offset zoals "+1"), max. 2 decimalen
for ax in (ax_geo, ax_tv):
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

# Waarschuwen als er waarden buiten de vaste assen vallen (die worden afgekapt)
for kolom, (lo, hi) in (("geo_d", GEO_YLIM), ("tv", TV_YLIM)):
    buiten = data[(data[kolom] < lo) | (data[kolom] > hi)]
    if not buiten.empty:
        print(
            f"Let op: {len(buiten)} waarde(n) van {kolom} vallen buiten de as {lo}-{hi} "
            f"(min {data[kolom].min():.3f}, max {data[kolom].max():.3f})."
        )

# Stagegrenzen en -labels
grenzen = data.index[data["layer"].diff().fillna(0) != 0].tolist()
for ax in (ax_geo, ax_tv):
    for g in grenzen:
        ax.axvline(g - 0.5, color="gray", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
for layer, groep in data.groupby("layer"):
    midden = groep.index.to_numpy().mean()
    soort = "full att." if layer == data["layer"].max() else "decomposed att."
    ax_geo.text(
        midden,
        1.02,
        f"stage {layer}\n{soort}",
        transform=ax_geo.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax_tv.set_xticks(x)
ax_tv.set_xticklabels(data["block"], fontsize=7)
ax_tv.set_xlabel("block (binnen stage)")
fig.suptitle(
    f"Geo-dieptegewicht en TV per block, epoch {epoch}, run {gekozen_run}", y=1.04
)

plt.tight_layout()
uitvoer = uitvoer or f"geo_tv_epoch{epoch}_{gekozen_run}.png"
plt.savefig(uitvoer, dpi=300, bbox_inches="tight")
print(f"Opgeslagen als {uitvoer}")
