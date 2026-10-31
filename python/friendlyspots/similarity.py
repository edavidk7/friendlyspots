import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Hexagonal Visium neighbors in array coordinates.
HEX_OFFSETS = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
METRICS = ("euclidean", "random_walk")


def spatial_similarity(samples, normalize=True, pools=None, metric="euclidean", unit_diagonal=False):
    """Spatial similarity between spatial transcriptomics samples.

    samples: dict of DataFrames with columns array_row, array_col (hexagonal
        grid array coordinates) and label.
    normalize: divide each sample's connection counts by their total.
    pools: optional dict mapping pool names to lists of sample names;
        connection counts are summed within each pool.
    metric: how connection counts are compared. "random_walk" gives
        absolute scores that don't depend on which other samples are
        compared, and requires normalize=True.
        - "euclidean": 1 - distance / max distance, so the least similar pair
          scores 0.
        - "random_walk": compares, for each label, the distribution of its
          neighbors' labels (sqrt Jensen-Shannon), averaged with weights by
          label share. Removes the effect of how common each label is. Each
          label gets one self loop, so a label absent from a sample always
          stays put there, and counts as different from its neighbors in a
          sample that has it.
    unit_diagonal: scale each sample's connection counts C (with one self
        loop per label added) to C_ij / sqrt(C_ii C_jj), so each label's
        connections with itself are 1, removing how much of each label there
        is. Makes rare labels count for more, so also their noise.
    Returns a symmetric DataFrame of similarities in [0, 1], or NaN for
    samples with no neighboring spots when normalize=True.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    if metric != "euclidean" and not normalize:
        raise ValueError(f"metric='{metric}' requires normalize=True")
    if any(s["label"].isna().any() for s in samples.values()):
        # e.g. pd.read_csv() reads a label literally called "NA" as missing
        warnings.warn(
            "Some labels are 'NA'. These will be treated as a label called "
            "'NA' and used in the comparison, rather than being treated as "
            "the absence of a label and dropped."
        )
    samples = {name: s.assign(label=s["label"].astype(str).where(s["label"].notna(), "NA")) for name, s in samples.items()}
    # Sorted as set order varies between runs, changing floating-point results
    labels = sorted(set().union(*(s["label"].astype(str) for s in samples.values())))
    graphs = {name: _graph(s, labels) for name, s in samples.items()}

    def pool(df):
        if pools is None:
            return df
        return pd.DataFrame({name: df.loc[members].sum() for name, members in pools.items()}).T

    counts = pool(pd.DataFrame({name: (x.T @ a @ x).toarray().ravel() for name, (x, a) in graphs.items()}).T)
    empty = (counts.sum(axis=1) == 0).to_numpy()
    # random_walk adds self loops; they must go on counts rather than proportions
    prepared = np.stack([
        _prepare_counts(c.reshape(len(labels), len(labels)), metric == "random_walk", unit_diagonal).ravel()
        for c in counts.to_numpy(dtype=float)
    ]) if len(counts) else counts.to_numpy(dtype=float)
    counts = pd.DataFrame(prepared, index=counts.index, columns=counts.columns)
    if normalize:
        if empty.any():
            warnings.warn(
                "Similarities are NA for samples with no neighboring spots: "
                + ", ".join(map(str, counts.index[empty]))
            )
        counts = counts.div(counts.sum(axis=1), axis=0)
        counts[empty] = np.nan
    x = counts.to_numpy(dtype=float)
    if metric == "random_walk":
        m = prepared.reshape(len(prepared), len(labels), len(labels))
        d = np.zeros((len(m), len(m)))
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                d[i, j] = d[j, i] = _random_walk_distance(m[i], m[j])
        d[empty, :] = d[:, empty] = np.nan
        return pd.DataFrame(1 - d, index=counts.index, columns=counts.index)
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    # Identical samples give a max distance of 0, so don't divide by it
    d_max = np.nanmax(d, initial=0)
    if d_max > 0:
        d = d / d_max
    return pd.DataFrame(1 - d, index=counts.index, columns=counts.index)


def _js(p, q):
    """Square root of the base-2 Jensen-Shannon divergence along the last axis, in [0, 1]."""
    m = (p + q) / 2
    # Empty bins contribute 0; masked so 0 * log(0) doesn't give NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        js = (np.where(p > 0, p * np.log2(p / m), 0) + np.where(q > 0, q * np.log2(q / m), 0)).sum(axis=-1) / 2
    # Rounding can make identical rows slightly negative
    return np.sqrt(np.maximum(js, 0))


def _prepare_counts(c, self_loops, unit_diagonal):
    """Connection counts with self loops, and scaled to a unit diagonal, as requested."""
    if self_loops or unit_diagonal:
        c = c + np.eye(len(c))
    if unit_diagonal:
        d = np.sqrt(np.diag(c))
        c = c / np.outer(d, d)
    return c


def _random_walk_distance(a, b):
    """Mean sqrt JS between labels' next-step distributions, weighted by label share.

    a and b are connection counts with one self loop per label already added
    (see _prepare_counts), which makes every row a distribution, so labels
    absent from a sample are absorbing states.
    """
    ra, rb = a.sum(axis=1), b.sum(axis=1)
    weights = (ra / ra.sum() + rb / rb.sum()) / 2
    return np.average(_js(a / ra[:, None], b / rb[:, None]), weights=weights)


def _onehot(codes, n_labels):
    n = len(codes)
    return sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, n_labels))


def _graph(sample, labels):
    """One-hot label matrix and hexagonal neighbor adjacency matrix of a sample's spots."""
    codes = pd.Categorical(sample["label"].astype(str), categories=labels).codes
    return _onehot(codes, len(labels)), _adjacency(sample)


def _adjacency(sample):
    """Hexagonal neighbor adjacency matrix of a sample's spots."""
    coords = list(zip(sample["array_row"], sample["array_col"]))
    index = {rc: i for i, rc in enumerate(coords)}
    rows, cols = [], []
    for i, (r, c) in enumerate(coords):
        for dr, dc in HEX_OFFSETS:
            j = index.get((r + dr, c + dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
    n = len(coords)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
