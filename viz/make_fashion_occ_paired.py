"""Twee gekoppelde datasets uit FashionMNIST: met en zonder occlusie.

    datasets/FashionOcc_<n>/     twee stukken overlappen (overlap binnen --overlap)
    datasets/FashionNoOcc_<n>/   dezelfde twee stukken, maar zonder overlap (masker-afstand >= --gap)

Per scène is ALLES gelijk tussen de twee datasets: klassenpaar, bronbeelden uit FashionMNIST,
schaal van elk stuk, achtergrondkleur, kledingkleur(en), diepte-waarden, en de split (train/test).
Alleen de positie van de stukken verschilt. Index i in beide mappen = dezelfde scène.

Formaat en labels zijn identiek aan make_fashion_occ.py (RGB/, Label/, Depth/, Region/, Rand/,
train.txt, test.txt, meta.csv). In FashionNoOcc is Rand/ leeg (geen occlusierand) en is Region 1
het volledige achterste stuk.

Let op: twee stukken moeten zonder overlap op het canvas passen. Daarom is de default --item_size
kleiner (0.40-0.55) dan in make_fashion_occ.py (0.55-0.75). Dezelfde grootte geldt voor beide datasets.

Zet dit bestand naast make_fashion_occ.py (helpers worden daaruit geïmporteerd).

Gebruik:
    python make_fashion_occ_paired.py --out datasets --sizes 100 200 500
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from make_fashion_occ import KLASSEN, KLEUREN, NAMEN, load_fashion, plaats, schaal_item

SUBS = ("RGB", "Label", "Depth", "Region", "Rand")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets")
    p.add_argument("--data_root", default="fashion_raw")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 500])
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--canvas", type=int, default=224)
    p.add_argument("--item_size", type=float, nargs=2, default=[0.40, 0.55])
    p.add_argument("--overlap", type=float, nargs=2, default=[0.25, 0.6])
    p.add_argument(
        "--gap",
        type=float,
        default=4,
        help="minimale afstand (px) tussen de stukken in de dataset zonder occlusie",
    )
    p.add_argument("--mask_thr", type=float, default=0.05)
    p.add_argument("--kleur", choices=["same", "different"], default="same")
    p.add_argument("--depth", choices=["informative", "constant"], default="informative")
    p.add_argument("--depth_front", type=int, default=1)
    p.add_argument("--depth_back", type=int, default=128)
    p.add_argument("--rand_px", type=float, default=10)
    p.add_argument("--max_pogingen", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def zoek_plaatsing(items, args, rng, met_occlusie):
    """Willekeurige posities tot aan de voorwaarde is voldaan. None als het niet lukt."""
    c = args.canvas
    for _ in range(args.max_pogingen):
        vlakken = []
        for it in items:
            h, w = it.shape
            y, x = rng.integers(0, c - h + 1), rng.integers(0, c - w + 1)
            vlakken.append(plaats(it, y, x, c))
        m_achter, m_voor = (v > args.mask_thr for v in vlakken)
        if met_occlusie:
            overlap = (m_achter & m_voor).sum() / max(m_achter.sum(), 1)
            if args.overlap[0] <= overlap <= args.overlap[1]:
                return vlakken
        else:
            dicht_bij_voor = distance_transform_edt(~m_voor) <= args.gap
            if not (m_achter & dicht_bij_voor).any():
                return vlakken
    return None


def render(vlakken, klassen, bg, kl_achter, kl_voor, args):
    """Zelfde rendering als make_fashion_occ.maak_scene."""
    c = args.canvas
    achter, voor = vlakken
    m_achter, m_voor = achter > args.mask_thr, voor > args.mask_thr

    rgb = np.empty((c, c, 3), np.float32)
    rgb[:] = KLEUREN[bg]
    label = np.zeros((c, c), np.uint8)
    depth = np.full((c, c), 254, np.uint8)
    for intens, masker, kleur, klasse, diepte in (
        (achter, m_achter, kl_achter, klassen[0], args.depth_back),
        (voor, m_voor, kl_voor, klassen[1], args.depth_front),
    ):
        textuur = 0.35 + 0.65 * intens[..., None]
        rgb[masker] = (np.array(KLEUREN[kleur], np.float32) * textuur)[masker]
        label[masker] = klasse + 1
        if args.depth == "informative":
            depth[masker] = diepte

    zicht_achter = m_achter & ~m_voor
    region = np.zeros((c, c), np.uint8)
    region[zicht_achter] = 1
    region[m_voor] = 2

    rand_bool = (zicht_achter & (distance_transform_edt(~m_voor) <= args.rand_px)) | (
        m_voor & (distance_transform_edt(~zicht_achter) <= args.rand_px)
    )
    rand = np.where(rand_bool, 250, 0).astype(np.uint8)

    overlap = (m_achter & m_voor).sum() / max(m_achter.sum(), 1)
    info = {
        "achtergrond": bg,
        "kleur_achter": kl_achter,
        "kleur_voor": kl_voor,
        "overlap": round(float(overlap), 3),
        "zichtbaar_achter": round(float(zicht_achter.sum() / max(m_achter.sum(), 1)), 3),
        "rand_pixels": int(rand_bool.sum()),
    }
    return rgb.round().astype(np.uint8), label, depth, region, rand, info


def maak_paar(klassen, imgs, idx_per_class, args, rng):
    """Eén set items + kleuren, twee plaatsingen: (scène met occlusie, scène zonder)."""
    while True:
        items = [schaal_item(imgs[rng.choice(idx_per_class[k])], args, rng) for k in klassen]
        occ = zoek_plaatsing(items, args, rng, met_occlusie=True)
        vrij = zoek_plaatsing(items, args, rng, met_occlusie=False)
        if occ is not None and vrij is not None:
            break  # anders: nieuwe items/groottes, geldt voor beide varianten

    bg = NAMEN[rng.integers(3)]
    andere = [n for n in NAMEN if n != bg]
    if args.kleur == "same":
        kl_achter = kl_voor = andere[rng.integers(2)]
    else:
        kl_achter, kl_voor = rng.permutation(andere)

    return (
        render(occ, klassen, bg, kl_achter, kl_voor, args),
        render(vrij, klassen, bg, kl_achter, kl_voor, args),
    )


def build_split(n, imgs, labels, args, rng):
    idx_per_class = [np.flatnonzero(labels == k) for k in range(10)]
    paren = [(k, (k + 1 + rng.integers(9)) % 10) for k in rng.permutation(10)]
    paren += [((k + 1 + rng.integers(9)) % 10, k) for k in rng.permutation(10)]
    while len(paren) < n:
        a, b = rng.choice(10, size=2, replace=False)
        paren.append((int(a), int(b)))
    return [(paar, *maak_paar(paar, imgs, idx_per_class, args, rng)) for paar in paren[:n]]


def save(samples, variant, root, start_idx, split):
    """variant 0 = met occlusie, 1 = zonder."""
    regels, meta = [], []
    for i, (paar, *scenes) in enumerate(samples, start=start_idx):
        achter, voor = paar
        rgb, label, depth, region, rand, info = scenes[variant]
        for sub, arr in zip(SUBS, (rgb, label, depth, region, rand)):
            Image.fromarray(arr).save(os.path.join(root, sub, f"{i}.png"))
        regels.append(f"RGB/{i}.png\tLabel/{i}.png")
        meta.append([
            i, split, KLASSEN[achter], achter + 1, KLASSEN[voor], voor + 1,
            info["achtergrond"], info["kleur_achter"], info["kleur_voor"],
            info["overlap"], info["zichtbaar_achter"], info["rand_pixels"],
        ])
    return regels, meta


def main():
    args = parse_args()
    if min(args.sizes) < 20:
        raise ValueError("Elke trainset moet >= 20 beelden hebben.")

    train_imgs, train_labels = load_fashion(args.data_root, train=True)
    test_imgs, test_labels = load_fashion(args.data_root, train=False)

    train_pool = build_split(max(args.sizes), train_imgs, train_labels, args,
                             np.random.default_rng(args.seed))
    test_pool = build_split(args.test_size, test_imgs, test_labels, args,
                            np.random.default_rng(args.seed + 1))

    for n in sorted(args.sizes):
        for variant, naam in enumerate(("FashionOcc", "FashionNoOcc")):
            root = os.path.join(args.out, f"{naam}_{n}")
            for sub in SUBS:
                os.makedirs(os.path.join(root, sub), exist_ok=True)

            tr_regels, tr_meta = save(train_pool[:n], variant, root, 0, "train")
            te_regels, te_meta = save(test_pool, variant, root, n, "test")
            for bestand, regels in (("train.txt", tr_regels), ("test.txt", te_regels)):
                with open(os.path.join(root, bestand), "w") as f:
                    f.write("\n".join(regels) + "\n")
            with open(os.path.join(root, "meta.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["idx", "split", "klasse_achter", "label_achter", "klasse_voor",
                            "label_voor", "achtergrond", "kleur_achter", "kleur_voor",
                            "overlap", "zichtbaar_achter", "rand_pixels"])
                w.writerows(tr_meta + te_meta)

            ov = np.array([m[9] for m in tr_meta])
            print(f"{naam}_{n}: overlap gem {ov.mean():.2f} (min {ov.min():.2f}, max {ov.max():.2f})")


if __name__ == "__main__":
    main()