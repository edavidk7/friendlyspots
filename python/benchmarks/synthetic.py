"""Synthetic benchmark: does a setting recover spatial arrangement independently of composition?

Each sample is a random tissue blob (size, shape and orientation vary) on the
Visium grid with 4 labels. Labels are assigned by ranking a score field into
bins, so each arrangement class can be generated with any composition:

    layers ABCD, layers ACBD (same composition, different neighbors), rings,
    smooth patches, voronoi patches (random neighbors), mixed (no structure)

Retrieval ground truth is the arrangement class, under four conditions:
fixed composition, composition varying between samples, and varying
composition with boundary or random label noise.

--alternatives also scores features that were tried and not added to the
package (WL hashes, PPMI, multi-scale walks without NPMI, the spectral gap of
the label graph, NPMI averaged over several walk lengths, random_walk on
heat-diffused contacts, and comparing where steps that leave a label land
together with the probability of staying, on direct or heat-diffused contacts).
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from common import SETTINGS, per_query, similarity, sqrt_js
from friendlyspots.similarity import _graph, _heat, _js, _random_walk_distance

HEX_OFFSETS = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
K = 4
BASE = np.array([0.4, 0.3, 0.2, 0.1])
CLASSES = ["layers ABCD", "layers ACBD", "rings", "smooth patches", "voronoi", "mixed"]
REPS = 5
CONDITIONS = {
    "fixed comp": (None, None),
    "varied comp": (10, None),
    "+ boundary noise 0.3": (10, ("boundary", 0.3)),
    "+ random noise 0.15": (10, ("random", 0.15)),
}


# ---- generator --------------------------------------------------------------

def tissue(rng):
    """Spots inside a random blob: an ellipse with a wobbly edge, randomly rotated."""
    radius = rng.uniform(15, 30)
    aspect, rot = rng.uniform(0.6, 1), rng.uniform(0, np.pi)
    wobble = [(m, rng.normal(0, 0.06), rng.uniform(0, 2 * np.pi)) for m in (2, 3, 4, 5)]
    n = int(radius * 1.6) + 4
    r, c = np.meshgrid(np.arange(-n, n + 1), np.arange(-2 * n, 2 * n + 1), indexing="ij")
    on_grid = (r + c) % 2 == 0
    r, c = r[on_grid], c[on_grid]
    x, y = c / 2, r * np.sqrt(3) / 2
    xr, yr = x * np.cos(rot) + y * np.sin(rot), (-x * np.sin(rot) + y * np.cos(rot)) / aspect
    theta = np.arctan2(yr, xr)
    inside = np.hypot(xr, yr) < radius * (1 + sum(a * np.cos(m * theta + ph) for m, a, ph in wobble))
    return r[inside], c[inside]


def adjacency(r, c):
    index = {rc: i for i, rc in enumerate(zip(r, c))}
    rows, cols = [], []
    for (a, b), i in index.items():
        for dr, dc in HEX_OFFSETS:
            j = index.get((a + dr, b + dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(r), len(r)))


def laplacian(a):
    return sp.diags(np.asarray(a.sum(axis=1)).ravel()) - a


def assign(score, comp, order, rng):
    """Rank spots by score and cut into bins; bin b gets label order[b], with its share comp[order[b]]."""
    cuts = np.round(np.cumsum(comp[order]) * len(score)).astype(int)
    ranks = np.argsort(np.argsort(score + 1e-9 * rng.standard_normal(len(score))))
    return np.array(order)[np.minimum(np.searchsorted(cuts, ranks, side="right"), K - 1)]


def arrange(cls, r, c, a, comp, rng):
    x, y = c / 2, r * np.sqrt(3) / 2
    if cls.startswith("layers"):
        t = rng.uniform(0, np.pi)
        return assign(x * np.cos(t) + y * np.sin(t), comp, [0, 1, 2, 3] if cls.endswith("ABCD") else [0, 2, 1, 3], rng)
    if cls == "rings":
        return assign(np.hypot(x - x.mean(), y - y.mean()), comp, [0, 1, 2, 3], rng)
    if cls == "smooth patches":
        return assign(expm_multiply(-15 * laplacian(a), rng.standard_normal(len(r))), comp, [0, 1, 2, 3], rng)
    if cls == "voronoi":
        seeds = rng.choice(len(r), max(4, len(r) // 60), replace=False)
        cell = np.argmin((x[:, None] - x[seeds]) ** 2 + (y[:, None] - y[seeds]) ** 2, axis=1)
        return assign(rng.permutation(len(seeds))[cell] + 0.5 * rng.random(len(r)), comp, list(rng.permutation(K)), rng)
    return assign(rng.random(len(r)), comp, [0, 1, 2, 3], rng)


def add_noise(label, a, kind, p, rng):
    label = label.copy()
    if kind == "random":
        flip = rng.random(len(label)) < p
        label[flip] = (label[flip] + rng.integers(1, K, flip.sum())) % K
        return label
    # Boundary spots take a random neighbor's label, as when a mixed spot is
    # assigned the wrong one of its labels
    new = label.copy()
    for i in range(len(label)):
        nb = a.indices[a.indptr[i]:a.indptr[i + 1]]
        if len(nb) and (label[nb] != label[i]).any() and rng.random() < p:
            new[i] = label[rng.choice(nb)]
    return new


def dataset(seed, comp_alpha, noise):
    rng = np.random.default_rng(seed)
    samples, truth = {}, {}
    for cls in CLASSES:
        for rep in range(REPS):
            r, c = tissue(rng)
            a = adjacency(r, c)
            comp = BASE if comp_alpha is None else rng.dirichlet(BASE * comp_alpha)
            comp = np.maximum(comp, 0.03)
            label = arrange(cls, r, c, a, comp / comp.sum(), rng)
            if noise:
                label = add_noise(label, a, *noise, rng)
            name = f"{cls} {rep}"
            samples[name] = pd.DataFrame({"array_row": r - r.min(), "array_col": c - c.min(), "label": label})
            truth[name] = cls
    # Shuffled so ties can't be broken by generation order
    order = rng.permutation(list(samples))
    return {n: samples[n] for n in order}, truth


# ---- alternatives tried ------------------------------------------------------

def walks(sample, k):
    """Label-by-label counts of walks of length k between spots."""
    a = adjacency(sample["array_row"].to_numpy(), sample["array_col"].to_numpy())
    x = np.eye(K)[sample["label"].to_numpy()]
    y = x
    for _ in range(k):
        y = a @ y
    return x.T @ y


def neighborhood_hashes(sample, as_set):
    """Histogram of (label, neighbor labels) per spot: 1-iteration Weisfeiler-Lehman."""
    a = adjacency(sample["array_row"].to_numpy(), sample["array_col"].to_numpy())
    label = sample["label"].to_numpy()
    hist = {}
    for i in range(len(label)):
        nb = label[a.indices[a.indptr[i]:a.indptr[i + 1]]]
        key = (label[i], frozenset(nb) if as_set else tuple(sorted(nb)))
        hist[key] = hist.get(key, 0) + 1
    return hist


def hist_distance(a, b):
    keys = sorted(set(a) | set(b), key=str)
    p, q = (np.array([h.get(k, 0) for k in keys], float) for h in (a, b))
    return sqrt_js(p / p.sum(), q / q.sum())


def npmi(m, positive=False):
    p = m / m.sum()
    marginal = p.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p / np.outer(marginal, marginal))
        if positive:
            return np.where(p > 0, np.maximum(pmi, 0), 0)
        return np.where(p > 0, pmi / -np.log2(p), -1)


def spectral_gap(m):
    d = m.sum(axis=1)
    keep = d > 0
    m, d = m[np.ix_(keep, keep)], d[keep]
    return 1 - np.sort(np.linalg.eigvalsh(m / np.sqrt(np.outer(d, d))))[-2]


def heat(sample, t):
    """Label-by-label heat after diffusing labels for time t, as npmi_heat uses."""
    onehot, a = _graph(sample.assign(label=sample["label"].astype(str)), [str(k) for k in range(K)])
    return _heat(onehot, a, t)


def exit_distance(a, b, stay=True):
    """Weighted mean sqrt JS between each shared label's exit distribution.

    A label's exit distribution is where a step from one of its spots lands,
    given that it leaves the label: its row of contacts without the diagonal.
    With stay, the probability of not leaving is compared too, as a
    Bernoulli distribution, averaged equally with the exit distance.
    """
    stays_a, stays_b = np.diag(a), np.diag(b)
    out_a, out_b = a.sum(axis=1) - stays_a, b.sum(axis=1) - stays_b
    shared = (out_a > 0) & (out_b > 0)
    if not shared.any():
        return 1.0
    off_a, off_b = a - np.diag(stays_a), b - np.diag(stays_b)
    d = _js(off_a[shared] / out_a[shared, None], off_b[shared] / out_b[shared, None])
    if stay:
        sa, sb = stays_a[shared] / a.sum(axis=1)[shared], stays_b[shared] / b.sum(axis=1)[shared]
        d = (d + _js(np.column_stack([sa, 1 - sa]), np.column_stack([sb, 1 - sb]))) / 2
    weights = (out_a / out_a.sum() + out_b / out_b.sum())[shared] / 2
    return np.average(d, weights=weights)


def contacts(sample):
    onehot, a = _graph(sample.assign(label=sample["label"].astype(str)), [str(k) for k in range(K)])
    return (onehot.T @ a @ onehot).toarray()


ALTERNATIVES = {
    "WL multiset hash": (lambda s: neighborhood_hashes(s, False), hist_distance),
    "WL set hash": (lambda s: neighborhood_hashes(s, True), hist_distance),
    "PPMI": (lambda s: walks(s, 1), lambda a, b: np.sqrt(np.mean((npmi(a, True) - npmi(b, True)) ** 2))),
    "walks k=8, JS": (lambda s: walks(s, 8), lambda a, b: sqrt_js(a.ravel() / a.sum(), b.ravel() / b.sum())),
    "spectral gap": (lambda s: spectral_gap(walks(s, 1)), lambda a, b: abs(a - b)),
    "NPMI mean of k=1,2,4,8": (lambda s: [walks(s, k) for k in (1, 2, 4, 8)],
                               lambda a, b: np.mean([np.sqrt(np.mean((npmi(x) - npmi(y)) ** 2)) for x, y in zip(a, b)])),
    "random_walk on heat t=1": (lambda s: heat(s, 1), _random_walk_distance),
    "random_walk on heat t=4": (lambda s: heat(s, 4), _random_walk_distance),
    "exit + stay": (contacts, exit_distance),
    "exit + stay, heat t=4": (lambda s: heat(s, 4), exit_distance),
}


def alternative_similarity(samples, feature, distance):
    f = {n: feature(s) for n, s in samples.items()}
    names = list(samples)
    d = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d[i, j] = d[j, i] = distance(f[names[i]], f[names[j]])
    return pd.DataFrame(-d, index=names, columns=names)


# ---- benchmark -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", type=int, default=10, help="random datasets per condition")
    parser.add_argument("--alternatives", action="store_true", help="also score alternatives not in the package")
    args = parser.parse_args()

    methods = {name: (lambda samples, s=setting: similarity(samples, s)) for name, setting in SETTINGS.items()}
    if args.alternatives:
        methods.update({name: (lambda samples, f=f, d=d: alternative_similarity(samples, f, d))
                        for name, (f, d) in ALTERNATIVES.items()})

    scores = {}  # (condition, method) -> per-dataset mAP; (condition, method, class) -> APs
    for ci, (condition, (alpha, noise)) in enumerate(CONDITIONS.items()):
        for seed in range(args.datasets):
            samples, truth = dataset(1000 * seed + ci, alpha, noise)
            for name, method in methods.items():
                queries = per_query(method(samples), truth, np.random.default_rng(seed))
                scores.setdefault((condition, name), []).append(np.mean([ap for _, ap in queries.values()]))
                for q, (_, ap) in queries.items():
                    scores.setdefault((condition, name, truth[q]), []).append(ap)
        print(f"done: {condition}", flush=True)

    print(f"\nmAP, mean (sd) over {args.datasets} datasets of {len(CLASSES)} classes x {REPS} samples; chance ~0.2")
    print(f"{'setting':24}" + "".join(f"{c:>22}" for c in CONDITIONS))
    for name in methods:
        print(f"{name:24}" + "".join(
            f"{np.mean(scores[c, name]):>15.2f} ({np.std(scores[c, name]):.2f})" for c in CONDITIONS))

    show = [m for m in ("euclidean (default)", "random_walk", "npmi_heat t=4", "npmi_heat t=16") if m in methods]
    print("\nper-class mAP, varied composition")
    print(f"{'class':16}" + "".join(f"{m:>22}" for m in show))
    for cls in CLASSES:
        print(f"{cls:16}" + "".join(f"{np.mean(scores['varied comp', m, cls]):22.2f}" for m in show))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
