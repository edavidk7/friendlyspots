"""Human dorsolateral prefrontal cortex, 12 Visium sections (Maynard et al. 2021).

Spots carry manual cortical layer annotations (L1-L6, white matter). Design:
3 donors x 2 positions 300 um apart x 2 adjacent sections 10 um apart.

1. Clean sections: retrieve the same donor, and the adjacent section.
2. Random label noise on the real sections.
3. Random disc crops (~40% of a section): crops' composition depends on where
   they fall, while the laminar arrangement stays; retrieve the same donor
   and the same section.

Needs data/dlpfc.csv.gz: run ./download_data.sh dlpfc first.
"""
import warnings

import numpy as np
import pandas as pd

from common import SETTINGS, chance, load_dataset, retrieval, similarity

def add_noise(sample, p, rng, labels):
    sample = sample.copy()
    flip = rng.random(len(sample)) < p
    sample.loc[flip, "label"] = [rng.choice([l for l in labels if l != x]) for x in sample["label"][flip]]
    return sample


def crops(samples, rng, per_section=4, share=0.4):
    out = {}
    for s, g in samples.items():
        x, y = g["array_col"].to_numpy() / 2, g["array_row"].to_numpy() * np.sqrt(3) / 2
        for k in range(per_section):
            c = rng.integers(len(g))
            d2 = (x - x[c]) ** 2 + (y - y[c]) ** 2
            out[f"{s}_{k}"] = g[d2 <= np.quantile(d2, share)]
    return out


def main():
    samples, meta = load_dataset("dlpfc")
    donor_of, pair_of, ids = meta["donor"].to_dict(), meta["pair"].to_dict(), list(samples)
    comp = pd.DataFrame({s: g["label"].value_counts(normalize=True) for s, g in samples.items()}).T.fillna(0)
    comp.insert(0, "donor", [donor_of[s] for s in comp.index])
    print("layer proportions per section\n" + comp.round(2).to_string() + "\n")

    print("1. clean sections (mAP): same donor / adjacent section")
    print(f"{'(chance)':22}{chance(ids, donor_of)[1]:>8.2f} / {chance(ids, pair_of)[1]:.2f}")
    for name, setting in SETTINGS.items():
        sim = similarity(samples, setting)
        print(f"{name:22}{retrieval(sim, donor_of)[1]:>8.2f} / {retrieval(sim, pair_of)[1]:.2f}")

    labels = sorted({l for g in samples.values() for l in g["label"]})
    print("\n2. random label noise (mAP, mean of 5 draws): same donor / adjacent section")
    print(f"{'setting':22}" + "".join(f"{f'p={p}':>16}" for p in (0.15, 0.3)))
    for name, setting in SETTINGS.items():
        cells = []
        for p in (0.15, 0.3):
            scores = []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                sim = similarity({s: add_noise(g, p, rng, labels) for s, g in samples.items()}, setting)
                scores.append((retrieval(sim, donor_of)[1], retrieval(sim, pair_of)[1]))
            d, a = np.mean(scores, axis=0)
            cells.append(f"{d:>9.2f} / {a:.2f}")
        print(f"{name:22}" + "".join(cells))

    print("\n3. random crops, 4 per section (mAP, mean (sd) of 5 draws): same donor / same section")
    results = {name: [] for name in SETTINGS}
    for seed in range(5):
        cropped = crops(samples, np.random.default_rng(100 + seed))
        donor = {k: donor_of[k.split("_")[0]] for k in cropped}
        section = {k: k.split("_")[0] for k in cropped}
        for name, setting in SETTINGS.items():
            sim = similarity(cropped, setting)
            results[name].append((retrieval(sim, donor)[1], retrieval(sim, section)[1]))
    print(f"{'(chance)':22}{chance(list(cropped), donor, 200)[1]:>8.2f} / {chance(list(cropped), section, 200)[1]:.2f}")
    for name, scores in results.items():
        (d, s), (dsd, ssd) = np.mean(scores, axis=0), np.std(scores, axis=0)
        print(f"{name:22}{d:>8.2f} / {s:.2f}   ({dsd:.2f} / {ssd:.2f})")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
