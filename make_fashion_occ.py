"""Occlusie-dataset uit FashionMNIST: twee kledingstukken per scène die elkaar deels overlappen,
op verschillende diepte. In DFormer-formaat (RGB/, Label/, Depth/, train.txt, test.txt).

Scène:   achtergrond (R/G/B) met daarop twee kledingstukken van verschillende klassen.
         Het voorste stuk bedekt een deel van het achterste (overlap gecontroleerd via --overlap).
Labels:  0 = achtergrond, 1..10 = klasse 0..9 (zichtbaar stuk per pixel; het voorste wint).
         -> num_classes = 11, gt_transform = False, background (ignore) = 255
Diepte:  achtergrond 254, achterste stuk --depth_back (128), voorste stuk --depth_front (1).
         --depth constant: overal 254 (geen diepte-informatie, als controleconditie).
Kleur:   --kleur same (default): beide stukken dezelfde kleur, zodat de grens tussen voor en achter
         in RGB alleen uit de textuur/vorm blijkt en diepte er echt toe doet.
         --kleur different: elk stuk een eigen kleur (dan is de grens ook in RGB zichtbaar).
         De kleding houdt zijn grijswaardetextuur (kleur x helderheid); nooit gelijk aan de achtergrond.

Klassen FashionMNIST: 0 T-shirt, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt,
                      7 Sneaker, 8 Bag, 9 Ankle boot

Gebruik:
    python make_fashion_occ.py --out datasets --sizes 100 200 500
    -> datasets/FashionOcc_100/ ... met meta.csv (klassen, kleuren, overlap, zichtbaar deel achterste stuk)
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image

KLEUREN = {"rood": (255, 0, 0), "groen": (0, 255, 0), "blauw": (0, 0, 255)}
NAMEN = list(KLEUREN)
KLASSEN = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets")
    p.add_argument("--data_root", default="fashion_raw", help="waar torchvision FashionMNIST downloadt/vindt")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 500])
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--canvas", type=int, default=224)
    p.add_argument("--item_size", type=float, nargs=2, default=[0.55, 0.75],
                   help="langste zijde van een stuk als fractie van het canvas (min max)")
    p.add_argument("--overlap", type=float, nargs=2, default=[0.25, 0.6],
                   help="fractie van het achterste stuk dat door het voorste bedekt wordt (min max)")
    p.add_argument("--mask_thr", type=float, default=0.05,
                   help="pixels met helderheid boven deze drempel horen bij het kledingstuk")
    p.add_argument("--kleur", choices=["same", "different"], default="same")
    p.add_argument("--depth", choices=["informative", "constant"], default="informative")
    p.add_argument("--depth_front", type=int, default=1)
    p.add_argument("--depth_back", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_fashion(root, train):
    from torchvision.datasets import FashionMNIST

    ds = FashionMNIST(root, train=train, download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def schaal_item(bron, args, rng):
    """Bijsnijden tot de inhoud en schalen (aspect behouden) naar een willekeurige grootte."""
    rijen, kolommen = np.flatnonzero(bron.any(axis=1)), np.flatnonzero(bron.any(axis=0))
    bron = bron[rijen[0]:rijen[-1] + 1, kolommen[0]:kolommen[-1] + 1]
    lang = rng.uniform(*args.item_size) * args.canvas
    schaal = lang / max(bron.shape)
    h, w = max(1, round(bron.shape[0] * schaal)), max(1, round(bron.shape[1] * schaal))
    return np.asarray(Image.fromarray(bron).resize((w, h), Image.BILINEAR), np.float32) / 255.0


def plaats(intensiteit, y, x, c):
    """Zet een item-uitsnede op een leeg canvas op positie (y, x)."""
    vol = np.zeros((c, c), np.float32)
    h, w = intensiteit.shape
    vol[y:y + h, x:x + w] = intensiteit
    return vol


def maak_scene(klassen, imgs, idx_per_class, args, rng, max_pogingen=500):
    """Twee stukken (achter, voor) met overlap binnen --overlap. Geeft rgb, label, depth, info."""
    c = args.canvas
    while True:
        items = [schaal_item(imgs[rng.choice(idx_per_class[k])], args, rng) for k in klassen]
        for _ in range(max_pogingen):
            vlakken = []
            for it in items:
                h, w = it.shape
                y, x = rng.integers(0, c - h + 1), rng.integers(0, c - w + 1)
                vlakken.append(plaats(it, y, x, c))
            achter, voor = vlakken
            m_achter, m_voor = achter > args.mask_thr, voor > args.mask_thr
            overlap = (m_achter & m_voor).sum() / max(m_achter.sum(), 1)
            if args.overlap[0] <= overlap <= args.overlap[1]:
                break
        else:
            continue  # met deze groottes lukt het niet: nieuwe items/groottes proberen
        break

    bg = NAMEN[rng.integers(3)]
    andere = [n for n in NAMEN if n != bg]
    if args.kleur == "same":
        kl_achter = kl_voor = andere[rng.integers(2)]
    else:
        kl_achter, kl_voor = rng.permutation(andere)

    rgb = np.empty((c, c, 3), np.float32)
    rgb[:] = KLEUREN[bg]
    label = np.zeros((c, c), np.uint8)
    depth = np.full((c, c), 254, np.uint8)

    # Achter eerst, dan voor eroverheen. Kledingkleur moduleren met de textuur (0.35..1.0).
    for intens, masker, kleur, klasse, diepte in (
        (achter, m_achter, kl_achter, klassen[0], args.depth_back),
        (voor, m_voor, kl_voor, klassen[1], args.depth_front),
    ):
        textuur = 0.35 + 0.65 * intens[..., None]
        rgb[masker] = (np.array(KLEUREN[kleur], np.float32) * textuur)[masker]
        label[masker] = klasse + 1
        if args.depth == "informative":
            depth[masker] = diepte

    zichtbaar_achter = (label == klassen[0] + 1).sum() / max(m_achter.sum(), 1)
    info = {"achtergrond": bg, "kleur_achter": kl_achter, "kleur_voor": kl_voor,
            "overlap": round(float(overlap), 3), "zichtbaar_achter": round(float(zichtbaar_achter), 3)}
    return rgb.round().astype(np.uint8), label, depth, info


def build_split(n, imgs, labels, args, rng):
    idx_per_class = [np.flatnonzero(labels == k) for k in range(10)]
    # Elke klasse komt voor als voor- én als achterstuk; daarna uniform, altijd twee verschillende klassen
    paren = [(k, (k + 1 + rng.integers(9)) % 10) for k in rng.permutation(10)]   # garantie achter
    paren += [((k + 1 + rng.integers(9)) % 10, k) for k in rng.permutation(10)]  # garantie voor
    while len(paren) < n:
        a, b = rng.choice(10, size=2, replace=False)
        paren.append((int(a), int(b)))
    return [(paar, maak_scene(paar, imgs, idx_per_class, args, rng)) for paar in paren[:n]]


def save(samples, root, start_idx, split):
    regels, meta = [], []
    for i, ((achter, voor), (rgb, label, depth, info)) in enumerate(samples, start=start_idx):
        Image.fromarray(rgb).save(os.path.join(root, "RGB", f"{i}.png"))
        Image.fromarray(label).save(os.path.join(root, "Label", f"{i}.png"))
        Image.fromarray(depth).save(os.path.join(root, "Depth", f"{i}.png"))
        regels.append(f"RGB/{i}.png\tLabel/{i}.png")
        meta.append([i, split, KLASSEN[achter], achter + 1, KLASSEN[voor], voor + 1,
                     info["achtergrond"], info["kleur_achter"], info["kleur_voor"],
                     info["overlap"], info["zichtbaar_achter"]])
    return regels, meta


def main():
    args = parse_args()
    if min(args.sizes) < 20:
        raise ValueError("Elke trainset moet >= 20 beelden hebben (elke klasse 1x voor en 1x achter).")

    train_imgs, train_labels = load_fashion(args.data_root, train=True)
    test_imgs, test_labels = load_fashion(args.data_root, train=False)

    # Genest: kleinere trainsets zijn prefixen van de grootste; testset identiek voor alle groottes
    train_pool = build_split(max(args.sizes), train_imgs, train_labels, args, np.random.default_rng(args.seed))
    test_pool = build_split(args.test_size, test_imgs, test_labels, args, np.random.default_rng(args.seed + 1))

    for n in sorted(args.sizes):
        root = os.path.join(args.out, f"FashionOcc_{n}")
        for sub in ("RGB", "Label", "Depth"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)

        tr_regels, tr_meta = save(train_pool[:n], root, 0, "train")
        te_regels, te_meta = save(test_pool, root, n, "test")
        for naam, regels in (("train.txt", tr_regels), ("test.txt", te_regels)):
            with open(os.path.join(root, naam), "w") as f:
                f.write("\n".join(regels) + "\n")
        with open(os.path.join(root, "meta.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "split", "klasse_achter", "label_achter", "klasse_voor", "label_voor",
                        "achtergrond", "kleur_achter", "kleur_voor", "overlap", "zichtbaar_achter"])
            w.writerows(tr_meta + te_meta)

        ov = np.array([m[9] for m in tr_meta])
        print(f"FashionOcc_{n}: overlap gemiddeld {ov.mean():.2f} (min {ov.min():.2f}, max {ov.max():.2f})")


if __name__ == "__main__":
    main()