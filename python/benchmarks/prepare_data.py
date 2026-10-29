"""Process raw downloads (see download_data.sh) into data/<dataset>.csv.gz.

Each output has one row per spot, with columns sample, array_row, array_col
and label, plus the sample's ground truth groups used by the benchmarks.

Usage: python prepare_data.py [dlpfc] [heart]   (default: both)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

# Maynard et al. (2021): 3 donors x 2 positions 300 um apart x 2 adjacent sections 10 um apart
DLPFC_IDS = ["151507", "151508", "151509", "151510", "151669", "151670",
             "151671", "151672", "151673", "151674", "151675", "151676"]

# Kuppe et al. (2022) samples from a single zone, grouped as in the paper.
# Samples spanning several zones (e.g. RZ_BZ) are left out, as their group
# isn't unambiguous from the name.
HEART_GROUPS = {
    "control_P1": "myogenic", "control_P7": "myogenic", "control_P8": "myogenic", "control_P17": "myogenic",
    "RZ_P3": "myogenic", "RZ_P6": "myogenic", "RZ_P9": "myogenic", "RZ_P11": "myogenic",
    "IZ_P3": "ischaemic", "IZ_P10": "ischaemic", "IZ_P15": "ischaemic", "IZ_P16": "ischaemic",
    "FZ_P14": "fibrotic", "FZ_P18": "fibrotic", "FZ_P20": "fibrotic",
}


def prepare_dlpfc():
    raw = DATA / "raw" / "dlpfc"
    layers = pd.read_csv(raw / "layer_map.tsv", sep="\t", header=None, names=["barcode", "sample", "label"], dtype=str)
    out = []
    for i, s in enumerate(DLPFC_IDS):
        pos = pd.read_csv(raw / f"{s}_positions.txt", header=None,
                          names=["barcode", "in_tissue", "array_row", "array_col", "px_row", "px_col"])
        spots = pos[pos["in_tissue"] == 1].merge(layers[layers["sample"] == s], on="barcode")
        donor = ["Br5292", "Br5595", "Br8100"][i // 4]
        out.append(spots[["sample", "array_row", "array_col", "label"]].assign(
            donor=donor, pair=f"{donor} position {(i % 4) // 2}"))
    return pd.concat(out)


def array_coords(xy):
    """Recover Visium array_row, array_col from pixel coordinates.

    Each neighbor step is matched to the nearest of the 6 hexagonal
    directions and coordinates are propagated outward from one spot, so small
    image distortions don't accumulate the way a single global fit would.
    """
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(xy).query(xy, k=7)
    is_nb = dist[:, 1:] < 1.3 * np.median(dist[:, 1])
    # The most common neighbor angle (folded onto half the circle) is one lattice direction
    disp = (xy[idx[:, 1:]] - xy[:, None])[is_nb]
    angle = np.mod(np.arctan2(disp[:, 1], disp[:, 0]), np.pi)
    hist, edges = np.histogram(angle, bins=180, range=(0, np.pi))
    a1 = np.median(angle[np.abs(angle - edges[hist.argmax()]) < np.pi / 36])
    # Directions at a1 + k * 60 degrees, as steps in array (row, col)
    steps = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]
    dirs = np.array([[np.cos(a1 + k * np.pi / 3), np.sin(a1 + k * np.pi / 3)] for k in range(6)])
    unset = np.iinfo(int).min
    rc = np.full((len(xy), 2), unset)
    for start in range(len(xy)):  # one pass per connected piece of tissue
        if rc[start, 0] != unset:
            continue
        rc[start] = (0, 2 * start * 1000)  # far apart, so separate pieces don't collide
        stack = [start]
        while stack:
            i = stack.pop()
            for j, nb in zip(idx[i, 1:], is_nb[i]):
                if not nb:
                    continue
                step = np.array(steps[np.argmax(dirs @ (xy[j] - xy[i]))])
                if rc[j, 0] == unset:
                    rc[j] = rc[i] + step
                    stack.append(j)
                elif (rc[j] != rc[i] + step).any():
                    raise ValueError("spots don't fit a hexagonal lattice")
    return rc[:, 0] - rc[:, 0].min(), rc[:, 1] - rc[:, 1].min()


def prepare_heart():
    import h5py

    out = []
    for name, group in HEART_GROUPS.items():
        # Spots are labelled by their dominant deconvolved cell type, as provided
        with h5py.File(DATA / "raw" / "heart" / f"Visium_{name}.h5ad", "r") as f:
            xy = f["obsm/X_spatial"][:]
            codes = f["obs/cell_type_original"][:]
            categories = f["obs/__categories/cell_type_original"][:].astype(str)
        row, col = array_coords(xy)
        label = np.where(codes >= 0, categories[np.maximum(codes, 0)], "NA")
        out.append(pd.DataFrame({"sample": name, "array_row": row, "array_col": col, "label": label, "group": group}))
    return pd.concat(out)


def main():
    for dataset in sys.argv[1:] or ["dlpfc", "heart"]:
        spots = {"dlpfc": prepare_dlpfc, "heart": prepare_heart}[dataset]()
        path = DATA / f"{dataset}.csv.gz"
        spots.to_csv(path, index=False)
        print(f"wrote {path.name}: {spots['sample'].nunique()} samples, {len(spots)} spots")


if __name__ == "__main__":
    main()
