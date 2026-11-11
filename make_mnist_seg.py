"""MNIST-segmentatiedataset (11 klassen) met onbalans over combinaties cijfer x achtergrondkleur x cijferkleur.

command => python make_mnist_seg.py --out datasets --mode shortcut [ --max_size ] --sizes 100 200 500

Eén cijfer per beeld. Een "combinatie" is (cijfer, achtergrondkleur, cijferkleur):
    10 cijfers x 3 achtergrondkleuren x 2 cijferkleuren (nooit gelijk aan de achtergrond) = 60 combinaties.

Twee modi voor de trainset (--mode):
  tail      long-tail over de 60 combinaties:  p(rang r) ∝ ratio^(-r/59), willekeurige rangorde.
            Test vooral "bekende vs. onbekende combinatie".
  shortcut  elk cijfer krijgt één dominant kleurpaar (achtergrond, cijferkleur) en verschijnt daarmee
            met kans --dominance; anders met een van de 5 andere paren (uniform). Cijfers zelf zijn
            uniform verdeeld. Er zijn maar 6 kleurparen voor 10 cijfers, dus elk paar is dominant
            voor 1 of 2 cijfers (zie dominant.csv): kleur voorspelt de klasse sterk, maar niet uniek.
Garantie (--garantie): 'combi' = elke combinatie minstens 1x, 'cijfer' = elk cijfer minstens 1x
(met zijn dominante paar), 'geen'. Default: combi bij tail, cijfer bij shortcut
(de combi-garantie zou bij kleine sets de shortcut verdunnen).
De groottes zijn genest (100 ⊂ 200 ⊂ 500).
Test:   uniform over alle 60 combinaties (elke combinatie even vaak), identiek voor alle groottes.
        Zo kun je per combinatie meten of het model slechter is op zeldzame kleurcombinaties.
meta.csv: per beeld split, cijfer, kleuren, rang, trainfrequentie van de combinatie, en (shortcut)
          of de kleurcombinatie het dominante paar van dat cijfer is (congruent) -> per-combinatie analyse.

Labels: 0 = achtergrond, 1..10 = cijfer 0..9   ->  num_classes = 11, gt_transform = False, background (ignore) = 255
Diepte: achtergrond 254, cijfer 1 [--depth informative]  of overal 254 [--depth constant]

Grootte: standaard willekeurig (--digit_min..--digit_max) op een willekeurige plek; met --max_size wordt
        het cijfer bijgesneden tot de inkt en zo groot mogelijk gecentreerd (--fill), zodat het in de
        grove stages (stage 4 = 7x7 tokens bij 224 px) zoveel mogelijk tokens beslaat.

Gebruik:
    python make_mnist_seg.py --out datasets --sizes 100 200 500
    python make_mnist_seg.py --out datasets --mode shortcut --max_size --sizes 100 200 500
"""

import argparse
import csv
import itertools
import os

import numpy as np
from PIL import Image

KLEUREN = {"rood": (255, 0, 0), "groen": (0, 255, 0), "blauw": (0, 0, 255)}
PAREN = [(bg, fg) for bg, fg in itertools.product(KLEUREN, KLEUREN) if bg != fg]  # 6 kleurparen
COMBIS = [(d, bg, fg) for d in range(10) for bg, fg in PAREN]  # 60
COMBI_IDX = {c: i for i, c in enumerate(COMBIS)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets")
    p.add_argument("--mnist_root", default="mnist_raw", help="waar torchvision MNIST downloadt/vindt")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 500])
    p.add_argument("--test_per_combi", type=int, default=5, help="testbeelden per combinatie (x60)")
    p.add_argument("--canvas", type=int, default=224)
    p.add_argument("--digit_min", type=int, default=56, help="min. cijfergrootte in pixels")
    p.add_argument("--digit_max", type=int, default=112, help="max. cijfergrootte in pixels")
    p.add_argument("--max_size", action="store_true",
                   help="cijfer zo groot mogelijk: bijgesneden tot de inktrand, geschaald (aspect behouden) "
                        "tot --fill x canvas en gecentreerd; negeert --digit_min/--digit_max")
    p.add_argument("--fill", type=float, default=0.95,
                   help="[--max_size] langste zijde van het cijfer als fractie van het canvas")
    p.add_argument("--mode", choices=["tail", "shortcut"], default="tail")
    p.add_argument("--ratio", type=float, default=50.0,
                   help="[tail] frequentie vaakste / zeldzaamste combinatie")
    p.add_argument("--dominance", type=float, default=0.9,
                   help="[shortcut] kans dat een cijfer met zijn dominante kleurpaar verschijnt")
    p.add_argument("--garantie", choices=["combi", "cijfer", "geen"], default=None,
                   help="default: combi bij tail, cijfer bij shortcut")
    p.add_argument("--depth", choices=["informative", "constant"], default="informative")
    p.add_argument("--ignore_edges", action="store_true",
                   help="anti-aliased randpixels (0.1 < intensiteit <= 0.5) krijgen label 255 (ignore)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_mnist(root, train):
    from torchvision.datasets import MNIST

    ds = MNIST(root, train=train, download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def render(combi, mnist_imgs, idx_per_class, args, rng):
    """Eén beeld met één cijfer op willekeurige grootte en positie."""
    digit, bg, fg = combi
    c = args.canvas
    bron = mnist_imgs[rng.choice(idx_per_class[digit])]

    if args.max_size:
        # Bijsnijden tot de inkt (MNIST heeft ~4 px lege rand), dan zo groot mogelijk en gecentreerd
        rijen, kolommen = np.flatnonzero(bron.any(axis=1)), np.flatnonzero(bron.any(axis=0))
        bron = bron[rijen[0]:rijen[-1] + 1, kolommen[0]:kolommen[-1] + 1]
        schaal = args.fill * c / max(bron.shape)
        sh, sw = max(1, round(bron.shape[0] * schaal)), max(1, round(bron.shape[1] * schaal))
        y, x = (c - sh) // 2, (c - sw) // 2
    else:
        sh = sw = int(rng.integers(args.digit_min, args.digit_max + 1))
        y, x = rng.integers(0, c - sh + 1, size=2)

    alpha = np.asarray(Image.fromarray(bron).resize((sw, sh), Image.BILINEAR), np.float32) / 255.0

    rgb = np.empty((c, c, 3), np.float32)
    rgb[:] = KLEUREN[bg]
    vlak = rgb[y:y + sh, x:x + sw]
    rgb[y:y + sh, x:x + sw] = alpha[..., None] * np.array(KLEUREN[fg], np.float32) + (1 - alpha[..., None]) * vlak

    label = np.zeros((c, c), np.uint8)
    depth = np.full((c, c), 254, np.uint8)
    vg = alpha > 0.5
    label[y:y + sh, x:x + sw][vg] = digit + 1
    if args.ignore_edges:
        label[y:y + sh, x:x + sw][(alpha > 0.1) & ~vg] = 255
    if args.depth == "informative":
        depth[y:y + sh, x:x + sw][vg] = 1
    return rgb.round().astype(np.uint8), label, depth


def save(samples, root, start_idx, split, rang, train_freq, dominant):
    regels, meta = [], []
    for i, (ci, (rgb, label, depth)) in enumerate(samples, start=start_idx):
        Image.fromarray(rgb).save(os.path.join(root, "RGB", f"{i}.png"))
        Image.fromarray(label).save(os.path.join(root, "Label", f"{i}.png"))
        Image.fromarray(depth).save(os.path.join(root, "Depth", f"{i}.png"))
        regels.append(f"RGB/{i}.png\tLabel/{i}.png")
        d, bg, fg = COMBIS[ci]
        congruent = "" if dominant is None else int(dominant[d] == (bg, fg))
        meta.append([i, split, d, d + 1, bg, fg, int(rang[ci]), int(train_freq[ci]), congruent])
    return regels, meta


def train_kansen(args, rng):
    """Kans per combinatie + rang (1 = vaakst) + dominant kleurpaar per cijfer (shortcut)."""
    n_combi = len(COMBIS)
    if args.mode == "tail":
        volgorde = rng.permutation(n_combi)          # volgorde[r] = combinatie-index met rang r
        rang = np.empty(n_combi, int)
        rang[volgorde] = np.arange(n_combi)
        probs = args.ratio ** (-rang / (n_combi - 1))
        return probs / probs.sum(), rang, None

    # shortcut: elk cijfer een dominant paar; 6 paren over 10 cijfers -> elk paar 1 of 2 cijfers
    cijfers = rng.permutation(10)
    dominant = {int(d): PAREN[i % len(PAREN)] for i, d in enumerate(cijfers)}
    probs = np.zeros(n_combi)
    for d in range(10):
        for paar in PAREN:
            kans = args.dominance if paar == dominant[d] else (1 - args.dominance) / (len(PAREN) - 1)
            probs[COMBI_IDX[(d, *paar)]] = kans / 10  # cijfers uniform
    rang = np.argsort(np.argsort(-probs, kind="stable"), kind="stable")
    return probs, rang, dominant


def garantie_combis(args, rng, dominant):
    """Combinaties die vooraan in de trainset komen (zitten dus in elke geneste grootte)."""
    if args.garantie == "combi":
        return rng.permutation(len(COMBIS))
    if args.garantie == "cijfer":
        if dominant is None:
            return np.array([COMBI_IDX[(d, *PAREN[rng.integers(len(PAREN))])] for d in rng.permutation(10)])
        return np.array([COMBI_IDX[(d, *dominant[d])] for d in rng.permutation(10)])
    return np.array([], dtype=int)


def main():
    args = parse_args()
    if args.garantie is None:
        args.garantie = "combi" if args.mode == "tail" else "cijfer"
    n_combi = len(COMBIS)

    rng = np.random.default_rng(args.seed)
    rng_test = np.random.default_rng(args.seed + 1)

    probs, rang, dominant = train_kansen(args, rng)
    garantie = garantie_combis(args, rng, dominant)
    if min(args.sizes) < len(garantie):
        raise ValueError(f"Elke trainset moet >= {len(garantie)} beelden hebben voor de garantie '{args.garantie}'.")
    if dominant is not None:
        print("Dominant kleurpaar per cijfer:", {d: f"{fg} op {bg}" for d, (bg, fg) in sorted(dominant.items())})

    train_imgs, train_labels = load_mnist(args.mnist_root, train=True)
    test_imgs, test_labels = load_mnist(args.mnist_root, train=False)
    idx_train = [np.flatnonzero(train_labels == k) for k in range(10)]
    idx_test = [np.flatnonzero(test_labels == k) for k in range(10)]

    # Train: eerst de garantie, daarna trekken uit de kansen; kleinere sets zijn prefixen
    n_max = max(args.sizes)
    train_combis = np.concatenate([garantie, rng.choice(n_combi, n_max - len(garantie), p=probs)]).astype(int)
    train_pool = [(ci, render(COMBIS[ci], train_imgs, idx_train, args, rng)) for ci in train_combis]

    # Test: elke combinatie even vaak
    test_combis = rng_test.permutation(np.repeat(np.arange(n_combi), args.test_per_combi))
    test_pool = [(ci, render(COMBIS[ci], test_imgs, idx_test, args, rng_test)) for ci in test_combis]

    for n in sorted(args.sizes):
        root = os.path.join(args.out, f"MNISTSeg_{n}")
        for sub in ("RGB", "Label", "Depth"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)

        train_freq = np.bincount(train_combis[:n], minlength=n_combi)
        tr_regels, tr_meta = save(train_pool[:n], root, 0, "train", rang, train_freq, dominant)
        te_regels, te_meta = save(test_pool, root, n, "test", rang, train_freq, dominant)

        for naam, regels in (("train.txt", tr_regels), ("test.txt", te_regels)):
            with open(os.path.join(root, naam), "w") as f:
                f.write("\n".join(regels) + "\n")
        with open(os.path.join(root, "meta.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "split", "cijfer", "label", "achtergrond", "cijferkleur",
                        "combi_rang", "combi_trainfreq", "congruent"])
            w.writerows(tr_meta + te_meta)
        if dominant is not None:
            with open(os.path.join(root, "dominant.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cijfer", "achtergrond", "cijferkleur"])
                w.writerows([d, bg, fg] for d, (bg, fg) in sorted(dominant.items()))

        top = np.argsort(-train_freq)[:3]
        print(f"MNISTSeg_{n}: combinatie-frequentie min {train_freq.min()}, max {train_freq.max()}; "
              f"vaakst: " + ", ".join(f"{COMBIS[c]}={train_freq[c]}" for c in top))


if __name__ == "__main__":
    main()