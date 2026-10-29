"""Pig esophagus graft time course (Durkin et al. 2026), the regression test data.

Prints pooled time point similarities for each setting, and how well each
setting retrieves sections from the same time point (unpooled).
"""
import warnings
from pathlib import Path

import pandas as pd

from common import SETTINGS, chance, retrieval, similarity

DATA = Path(__file__).parents[1] / "tests" / "pig_esophagus_spots.csv.xz"
POOLS = {
    "Pre-implant": ["Pre-implant"],
    "1 month": ["1 month (D)"],
    "3 months": ["3 months (E)", "3 months (F)"],
    "6 months": ["6 months (B)", "6 months (C)"],
    "Native": ["Native DP1", "Native DP2", "Native DP3", "Native (C)"],
}


def main():
    spots = pd.read_csv(DATA)
    samples = {name: s for name, s in spots.groupby("sample_name")}
    group = {member: pool for pool, members in POOLS.items() for member in members}

    for name, setting in SETTINGS.items():
        if setting is not None:
            sim = similarity(samples, setting, pools=POOLS)
            print(f"{name}, pooled\n{sim.round(2).to_string()}\n")

    nn, mAP = chance(list(samples), group)
    print("retrieving sections from the same time point (unpooled; 8 queries)")
    print(f"{'setting':22}{'1-NN':>6}{'mAP':>7}")
    print(f"{'(chance)':22}{nn:6.2f}{mAP:7.2f}")
    for name, setting in SETTINGS.items():
        nn, mAP = retrieval(similarity(samples, setting), group)
        print(f"{name:22}{nn:6.2f}{mAP:7.2f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
