import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

# Hexagonal Visium neighbors in array coordinates.
HEX_OFFSETS = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
METRICS = ("euclidean", "random_walk", "npmi_heat")


def spatial_similarity(samples, normalize=True, pools=None, metric="euclidean", t=None):
    """Spatial similarity between spatial transcriptomics samples.

    samples: dict of DataFrames with columns array_row, array_col (hexagonal
        grid array coordinates) and label.
    normalize: divide each sample's connection counts by their total.
    pools: optional dict mapping pool names to lists of sample names;
        connection counts are summed within each pool.
    metric: how connection counts are compared. All metrics except
        "euclidean" give absolute scores that don't depend on which other
        samples are compared, and require normalize=True.
        - "euclidean": 1 - distance / max distance, so the least similar pair
          scores 0.
        - "random_walk": compares, for each label, the distribution of its
          neighbors' labels (sqrt Jensen-Shannon), averaged with weights by
          label share. Removes the effect of how common each label is.
        - "npmi_heat": compares normalized pointwise mutual information of
          label pairs, i.e. how much more or less often two labels are near
          each other than their abundance predicts, with each pair weighted
          by how common the two labels are. Removes the effect of
          composition. "Near" means labels diffused over the spot graph for
          time t, a neighborhood of roughly sqrt(3 * t) spots, which makes it
          robust to scattered mislabelled spots.
        "random_walk" and "npmi_heat" compare the neighbors of labels present
        in both samples, including neighbors only one sample has; samples
        with no labels in common score 0.
    t: diffusion time for metric="npmi_heat" (default 4).
    Returns a symmetric DataFrame of similarities in [0, 1], or NaN for
    samples with no neighboring spots when normalize=True.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    if metric != "euclidean" and not normalize:
        raise ValueError(f"metric='{metric}' compares proportions, so requires normalize=True")
    if t is not None and metric != "npmi_heat":
        raise ValueError("t is only used with metric='npmi_heat'")
    if metric == "npmi_heat":
        t = 4 if t is None else t
        if t <= 0:
            raise ValueError("t must be positive")
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
    if normalize:
        totals = counts.sum(axis=1)
        empty = (totals == 0).to_numpy()
        if empty.any():
            warnings.warn(
                "Similarities are NA for samples with no neighboring spots: "
                + ", ".join(map(str, counts.index[empty]))
            )
        counts = counts.div(totals, axis=0)
    x = counts.to_numpy(dtype=float)
    if metric in ("random_walk", "npmi_heat"):
        if metric == "npmi_heat":
            x = pool(pd.DataFrame({name: _heat(onehot, a, t).ravel() for name, (onehot, a) in graphs.items()}).T).to_numpy()
        m = x.reshape(len(x), len(labels), len(labels))
        dist = _random_walk_distance if metric == "random_walk" else _npmi_distance
        d = np.zeros((len(m), len(m)))
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                d[i, j] = d[j, i] = dist(m[i], m[j])
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


def _random_walk_distance(a, b):
    ra, rb = a.sum(axis=1), b.sum(axis=1)
    shared = (ra > 0) & (rb > 0)
    if not shared.any():
        return 1.0
    weights = (ra / ra.sum() + rb / rb.sum())[shared] / 2
    return np.average(_js(a[shared] / ra[shared, None], b[shared] / rb[shared, None]), weights=weights)


def _npmi(m):
    p = m / m.sum()
    marginal = p.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = np.log2(p / np.outer(marginal, marginal)) / -np.log2(p)
    # Labels that never touch get -1; a label touching only itself (p = 1) gets 1
    return np.where(p == 0, -1, np.where(p == 1, 1, npmi))


def _npmi_distance(a, b):
    """Root mean square NPMI difference, halved to lie in [0, 1].

    Rows are labels present in both samples; columns are labels present in
    either, so a shared label bordering a label the other sample lacks counts
    as a difference (-1 in the other sample). Each label pair is weighted by
    the product of the two labels' mean shares, as NPMI of rare labels is
    noisy and would otherwise count as much as that of common ones.
    """
    share_a, share_b = a.sum(axis=1) / a.sum(), b.sum(axis=1) / b.sum()
    rows, cols = (share_a > 0) & (share_b > 0), (share_a > 0) | (share_b > 0)
    if not rows.any():
        return 1.0
    share = (share_a + share_b) / 2
    weights = np.outer(share[rows], share[cols])
    diff = (_npmi(a) - _npmi(b))[np.ix_(rows, cols)]
    return np.sqrt((weights * diff**2).sum() / weights.sum()) / 2


def _heat(onehot, adjacency, t):
    """Label-by-label heat after diffusing each label over the spot graph for time t."""
    laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    # Clipped as expm_multiply can return tiny negative values
    return np.maximum(onehot.T @ expm_multiply(-t * laplacian, onehot.toarray()), 0)


def _graph(sample, labels):
    """One-hot label matrix and hexagonal neighbor adjacency matrix of a sample's spots."""
    codes = pd.Categorical(sample["label"].astype(str), categories=labels).codes
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
    onehot = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(labels)))
    adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return onehot, adjacency
