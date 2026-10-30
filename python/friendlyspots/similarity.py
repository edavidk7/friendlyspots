import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow
from scipy.sparse.linalg import expm_multiply

# Hexagonal Visium neighbors in array coordinates.
HEX_OFFSETS = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
METRICS = ("euclidean", "random_walk", "npmi_heat", "spectral")
# Graph frequency bands for metric="spectral", in units of lambda * N (N spots),
# as the frequency of a domain of a given share of the tissue scales as 1 / N
SPECTRAL_EDGES = np.concatenate([[0], np.geomspace(1, 1e4, 15)])
N_MOMENTS = 1000


def spatial_similarity(samples, normalize=True, pools=None, metric="euclidean", t=None, smoothing=0.2):
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
          label share. Removes the effect of how common each label is. Each
          label gets one self loop, so a label absent from a sample always
          stays put there, and counts as different from its neighbors in a
          sample that has it.
        - "npmi_heat": compares normalized pointwise mutual information of
          label pairs, i.e. how much more or less often two labels are near
          each other than their abundance predicts, with each pair weighted
          by how common the two labels are. Removes the effect of
          composition. "Near" means labels diffused over the spot graph for
          time t, a neighborhood of roughly sqrt(3 * t) spots, which makes it
          robust to scattered mislabelled spots. Compares the neighbors of
          labels present in both samples, including neighbors only one sample
          has; samples with no labels in common score 0.
        - "spectral": averages "random_walk" with a comparison of the
          labels' graph Fourier spectra: how each label's spatial pattern
          divides between large domains and fine-grained mixing, and at which
          scales pairs of labels occupy the same or complementary places.
          Captures structure beyond direct neighbors, and is robust to
          scattered mislabelled spots.
    t: diffusion time for metric="npmi_heat" (default 4).
    smoothing: lam passed to denoise_labels() to relabel isolated, likely
        mislabelled spots in each sample first. The default relabels isolated
        spots and pairs; 0 turns it off, e.g. to reproduce results computed
        without it.
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
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
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
    if smoothing > 0:
        graphs = {name: (_onehot(_smooth(x.indices, a, smoothing), len(labels)), a) for name, (x, a) in graphs.items()}

    def pool(df):
        if pools is None:
            return df
        return pd.DataFrame({name: df.loc[members].sum() for name, members in pools.items()}).T

    counts = pool(pd.DataFrame({name: (x.T @ a @ x).toarray().ravel() for name, (x, a) in graphs.items()}).T)
    raw = counts.to_numpy(dtype=float)
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
    if metric != "euclidean":
        if metric == "npmi_heat":
            m = pool(pd.DataFrame({name: _heat(onehot, a, t).ravel() for name, (onehot, a) in graphs.items()}).T).to_numpy()
            m = m.reshape(len(m), len(labels), len(labels))
            dist = lambda i, j: _npmi_distance(m[i], m[j])
        else:
            # Self loops are added to counts, so they must be raw rather than proportions
            m = raw.reshape(len(raw), len(labels), len(labels))
            dist = lambda i, j: _random_walk_distance(m[i], m[j])
        if metric == "spectral":
            moments = pool(pd.DataFrame({name: _chebyshev_moments(onehot, a).ravel() for name, (onehot, a) in graphs.items()}).T)
            sizes = pool(pd.DataFrame({name: np.asarray(onehot.sum(axis=0)).ravel() for name, (onehot, _) in graphs.items()}).T)
            moments = moments.to_numpy().reshape(len(moments), N_MOMENTS, len(labels), len(labels))
            sizes = sizes.to_numpy(dtype=float)
            spectra = [_bands(mu, np.minimum(SPECTRAL_EDGES / max(n.sum(), 1), 2)) for mu, n in zip(moments, sizes)]
            walk = dist
            dist = lambda i, j: (walk(i, j) + _spectral_distance(spectra[i], spectra[j], sizes[i], sizes[j])) / 2
        d = np.zeros((len(m), len(m)))
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                d[i, j] = d[j, i] = dist(i, j)
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
    """Mean sqrt JS between labels' next-step distributions, weighted by label share.

    a and b are raw connection counts. One self loop per label makes every
    row a distribution, so labels absent from a sample are absorbing states.
    """
    a, b = a + np.eye(len(a)), b + np.eye(len(b))
    ra, rb = a.sum(axis=1), b.sum(axis=1)
    weights = (ra / ra.sum() + rb / rb.sum()) / 2
    return np.average(_js(a / ra[:, None], b / rb[:, None]), weights=weights)


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


def _chebyshev_moments(onehot, adjacency, k=N_MOMENTS):
    """Jackson-damped Chebyshev moments of label signals on the normalized Laplacian.

    mu[k, c, c'] = g_k y_c^T T_k(L - I) y_c', where L = I - D^-1/2 A D^-1/2,
    y_c indicates spots of label c with its zero-frequency part (one mode per
    connected piece) removed, and g_k is the Jackson damping.
    """
    deg = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1)), 0)
    m = -(sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)).tocsr()  # L - I, spectrum in [-1, 1]
    y = onehot.toarray()
    # The lambda = 0 eigenvector of each piece is proportional to sqrt(degree)
    _, piece = connected_components(adjacency, directed=False)
    u0 = np.where(deg > 0, np.sqrt(deg), 1.0)
    for p in np.unique(piece):
        u = u0 * (piece == p)
        y -= np.outer(u, (u @ y) / (u @ u))
    mu = np.zeros((k, y.shape[1], y.shape[1]))
    t_prev, t_cur = y, m @ y
    mu[0], mu[1] = y.T @ t_prev, y.T @ t_cur
    for i in range(2, k):
        t_prev, t_cur = t_cur, 2 * (m @ t_cur) - t_prev
        mu[i] = y.T @ t_cur
    n = np.arange(k)
    jackson = ((k - n + 1) * np.cos(np.pi * n / (k + 1)) + np.sin(np.pi * n / (k + 1)) / np.tan(np.pi / (k + 1))) / (k + 1)
    return jackson[:, None, None] * mu


def _bands(mu, edges):
    """Energy E[b, c, c'] of the damped Chebyshev expansion in each band of lambda between edges.

    With lambda - 1 = cos(theta), the expansion's density integrates to
    g_0 mu_0 (theta_lo - theta_hi) / pi + sum_k 2 g_k mu_k (sin k theta_lo - sin k theta_hi) / (k pi).
    """
    n = np.arange(1, len(mu))
    theta = np.arccos(np.clip(np.asarray(edges) - 1, -1, 1))
    w = np.empty((len(edges) - 1, len(mu)))
    w[:, 0] = (theta[:-1] - theta[1:]) / np.pi
    w[:, 1:] = 2 * (np.sin(n * theta[:-1, None]) - np.sin(n * theta[1:, None])) / (n * np.pi)
    return np.einsum("bk,kcd->bcd", w, mu)


def _spectral_distance(ea, eb, count_a, count_b):
    """Distance between labels' spectrum shapes (sqrt JS) and pairs' coherence spectra, in [0, 1].

    A label's spectrum shape is its energy per band over its total. The
    coherence of labels c, c' in band b is E[b, c, c'] / sqrt(E[b, c, c] E[b, c', c']),
    in [-1, 1]. Labels and pairs are weighted by label share; a label present
    in one sample only differs maximally.
    """
    share_a, share_b = count_a / count_a.sum(), count_b / count_b.sum()
    # Clipped as the expansion can ripple slightly below 0
    auto_a = np.maximum(np.einsum("bcc->bc", ea), 1e-12)
    auto_b = np.maximum(np.einsum("bcc->bc", eb), 1e-12)
    both = (share_a > 0) & (share_b > 0)
    w = (share_a + share_b) / 2
    d_label = np.where(both, _js((auto_a / auto_a.sum(axis=0)).T, (auto_b / auto_b.sum(axis=0)).T), 1.0)
    d_shape = np.sum(w * d_label) / w.sum()
    pair_w = np.triu(np.outer(w * both, w * both), k=1)
    if pair_w.sum() == 0:
        return d_shape
    coh_a = np.clip(ea / np.sqrt(auto_a[:, :, None] * auto_a[:, None, :]), -1, 1)
    coh_b = np.clip(eb / np.sqrt(auto_b[:, :, None] * auto_b[:, None, :]), -1, 1)
    d_pair = np.abs(coh_a - coh_b).mean(axis=0) / 2
    return (d_shape + np.sum(pair_w * d_pair) / pair_w.sum()) / 2


def _heat(onehot, adjacency, t):
    """Label-by-label heat after diffusing each label over the spot graph for time t."""
    laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    # Clipped as expm_multiply can return tiny negative values
    return np.maximum(onehot.T @ expm_multiply(-t * laplacian, onehot.toarray()), 0)


def denoise_labels(sample, lam=0.2):
    """Relabel isolated, likely mislabelled spots.

    sample: DataFrame with columns array_row, array_col (hexagonal grid array
        coordinates) and label.
    lam: cost of each edge between touching spots with different labels,
        relative to a cost of 1 for changing a spot's label. A group of k
        spots with P edges to other labels is relabelled when lam * P > k:
        values below 1/6 change nothing, 0.2 removes isolated spots and
        pairs, and larger values remove larger groups and smooth boundaries,
        also erasing labels that are genuinely scattered.
    Returns a copy of sample with labels minimizing the number of changed
    labels plus lam times the number of edges between different labels
    (Potts model), found with alpha-expansion graph cuts.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    codes, uniques = pd.factorize(sample["label"], use_na_sentinel=False)
    smoothed = _smooth(codes, _adjacency(sample), lam)
    return sample.assign(label=pd.Series(uniques.take(smoothed), index=sample.index))


def _smooth(codes, adjacency, lam, max_iter=5):
    """Label codes minimizing sum [new != codes] + lam * sum over edges [new_p != new_q]."""
    if lam == 0 or len(codes) == 0:
        return codes
    upper = sp.triu(adjacency, k=1).tocoo()
    edges = (upper.row, upper.col)
    x = np.array(codes)
    # Expanding a label the sample doesn't have can't lower the energy
    for _ in range(max_iter):
        changed = False
        for alpha in np.unique(codes):
            switch = _expansion_move(x, codes, alpha, edges, lam) & (x != alpha)
            if switch.any():
                x[switch] = alpha
                changed = True
        if not changed:
            break
    return x


def _expansion_move(x, y, alpha, edges, lam, scale=1000):
    """Spots that should switch to label alpha in one alpha-expansion move (Boykov et al. 2001).

    Each spot keeps its label (z = 0) or switches to alpha (z = 1), chosen by
    an s-t min cut; spots left on the sink side switch. Costs are scaled to
    integers for scipy's max flow.
    """
    n = len(x)
    u0, u1 = (x != y).astype(float), (alpha != y).astype(float)  # unary costs of z = 0 and z = 1
    p, q = edges
    # Potts pairwise costs E(z_p, z_q), with E(1, 1) = 0, written as
    # E(0, 0) + (E(1, 0) - E(0, 0)) z_p - E(1, 0) z_q + (E(0, 1) + E(1, 0) - E(0, 0)) z_q (1 - z_p)
    e00, e01, e10 = lam * (x[p] != x[q]), lam * (x[p] != alpha), lam * (alpha != x[q])
    np.add.at(u1, p, e10 - e00)
    np.add.at(u1, q, -e10)
    u0 += np.maximum(-u1, 0)  # a negative cost of z = 1 is a cost of z = 0
    u1 = np.maximum(u1, 0)
    s, t = n, n + 1
    rows = np.concatenate([np.full(n, s), np.arange(n), p])
    cols = np.concatenate([np.arange(n), np.full(n, t), q])
    # The pairwise capacity is >= 0 as the Potts cost is a metric
    caps = np.round(np.concatenate([u1, u0, e01 + e10 - e00]) * scale).astype(np.int32)
    keep = caps > 0
    g = sp.csr_matrix((caps[keep], (rows[keep], cols[keep])), shape=(n + 2, n + 2))
    residual = (g - maximum_flow(g, s, t).flow).tocsr()
    residual.data = np.maximum(residual.data, 0)
    residual.eliminate_zeros()
    source_side = np.zeros(n + 2, bool)
    source_side[breadth_first_order(residual, s, return_predecessors=False)] = True
    return ~source_side[:n]


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
