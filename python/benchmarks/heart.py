"""Human myocardial infarction Visium sections (Kuppe et al. 2022, Nature).

Spots are labelled by their dominant deconvolved cell type. Ground truth is
each sample's patient group as in the paper: myogenic (control, remote zone),
ischaemic (ischaemic zone) or fibrotic (fibrotic zone); see prepare_data.py.

Needs data/heart.csv.gz: run ./download_data.sh heart first.
"""
import warnings

import pandas as pd

from common import SETTINGS, chance, load_dataset, retrieval, similarity


def main():
    samples, meta = load_dataset("heart")
    group = meta["group"].to_dict()
    comp = pd.DataFrame({n: s["label"].value_counts(normalize=True) for n, s in samples.items()}).T.fillna(0)
    comp.insert(0, "spots", [len(s) for s in samples.values()])
    comp.insert(0, "group", [group[n] for n in comp.index])
    print("dominant cell type proportions per section\n" + comp.round(2).to_string() + "\n")

    nn, mAP = chance(list(samples), group)
    print(f"{'setting':22}{'1-NN':>6}{'mAP':>7}")
    print(f"{'(chance)':22}{nn:6.2f}{mAP:7.2f}")
    for name, setting in SETTINGS.items():
        nn, mAP = retrieval(similarity(samples, setting), group)
        print(f"{name:22}{nn:6.2f}{mAP:7.2f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
