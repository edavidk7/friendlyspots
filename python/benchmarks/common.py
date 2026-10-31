"""Helpers shared by the benchmarks: settings compared, a composition-only
baseline, retrieval scoring and loading processed datasets."""
from pathlib import Path

import numpy as np
import pandas as pd

from friendlyspots import spatial_similarity

DATA = Path(__file__).parent / "data"

# None is the composition-only baseline; dicts are spatial_similarity() arguments
SETTINGS = {
    "composition only": None,
    "euclidean (default)": {},
    "random_walk": {"metric": "random_walk"},
    "rw, unit diagonal": {"metric": "random_walk", "unit_diagonal": True},
}


def sqrt_js(p, q):
    """Square root of the base-2 Jensen-Shannon divergence along the last axis."""
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        js = (np.where(p > 0, p * np.log2(p / m), 0) + np.where(q > 0, q * np.log2(q / m), 0)).sum(axis=-1) / 2
    return np.sqrt(np.maximum(js, 0))


def composition_similarity(samples):
    """1 - sqrt JS between label proportions, ignoring where spots are."""
    counts = pd.DataFrame({name: s["label"].value_counts() for name, s in samples.items()}).T.fillna(0)
    p = counts.div(counts.sum(axis=1), axis=0).to_numpy()
    return pd.DataFrame(1 - sqrt_js(p[:, None], p[None]), index=counts.index, columns=counts.index)


def similarity(samples, setting, **extra):
    """Similarities under a setting; extra arguments (e.g. pools) go to spatial_similarity()."""
    if setting is None:
        return composition_similarity(samples)
    return spatial_similarity(samples, **setting, **extra)


def per_query(sim, truth, rng=None):
    """For each query, whether its nearest neighbor is in its group, and its average precision.

    Queries are samples with at least one other sample in their group. Ties
    are broken randomly, so they can't favor the order samples were given in.
    """
    rng = rng or np.random.default_rng(0)
    names = list(sim.index)
    sim = sim.to_numpy() + 1e-12 * rng.random((len(names), len(names)))
    group = np.array([truth[n] for n in names])
    out = {}
    for q in range(len(names)):
        if (group == group[q]).sum() < 2:
            continue
        order = [j for j in np.argsort(-sim[q]) if j != q]
        relevant = group[order] == group[q]
        out[names[q]] = relevant[0], (np.cumsum(relevant) / np.arange(1, len(order) + 1))[relevant].mean()
    return out


def retrieval(sim, truth, rng=None):
    """Nearest-neighbor accuracy and mean average precision of retrieving same-group samples."""
    return tuple(np.mean(list(per_query(sim, truth, rng).values()), axis=0))


def chance(names, truth, reps=1000, seed=0):
    """Retrieval scores of random similarities, for reference."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(reps):
        m = rng.random((len(names), len(names)))
        scores.append(retrieval(pd.DataFrame((m + m.T) / 2, index=names, columns=names), truth, rng))
    return tuple(np.mean(scores, axis=0))


def load_dataset(name):
    """Samples and per-sample metadata of a dataset processed by prepare_data.py."""
    # Labels are read verbatim, so one called "NA" stays a label rather than missing
    spots = pd.read_csv(DATA / f"{name}.csv.gz", dtype={"sample": str}, keep_default_na=False)
    samples = {s: g[["array_row", "array_col", "label"]] for s, g in spots.groupby("sample", sort=False)}
    meta = spots.drop(columns=["array_row", "array_col", "label"]).drop_duplicates("sample").set_index("sample")
    return samples, meta
