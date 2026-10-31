"""Detection benchmark: does a method flag small biological alterations, but not noise?

For each base section, two noisy copies are compared (a negative pair), and
the same, but with an alteration of k spots in one copy (a positive pair).
A method's AUC is the probability that it gives a positive pair a larger
distance than a negative pair: 0.5 means it can't tell them apart, 1 that it
always ranks a real alteration above noise.

Every base section gets a rare cell type, scattered at random over 2% of its
spots (at least 40). Alterations, each a compact patch of k spots grown from a
center spot:
- niche cluster: k of the scattered rare spots move into a cluster at the
  interface between the section's two most common labels, so the rare type's
  count is unchanged and only its arrangement differs.
- new domain: an island of an existing label inside a region of a label it
  doesn't touch in that section (or touches least), i.e. an ectopic domain.

Noise, applied independently to both sections of a pair:
- random flips: each spot's label is replaced by another of the section's
  labels with probability p.
- boundary jitter: each spot bordering a different label takes a random
  neighbor's label with probability p, as when mixed spots are mislabelled.

Bases: 20 synthetic sections (layers, rings, smooth and Voronoi patches) and
the pig, cortex and heart sections; cortex and heart need ./download_data.sh.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from common import composition_similarity, load_dataset
from friendlyspots import spatial_similarity
from friendlyspots.similarity import _adjacency, _graph

import pig
import synthetic

NOISES = {"random flips 5%": ("random", 0.05), "random flips 10%": ("random", 0.10),
          "boundary jitter 30%": ("boundary", 0.30)}
ALTERATIONS = ["niche cluster", "new domain"]
SIZES = [5, 10, 20, 40]
RARE = "(rare)"


# ---- bases ------------------------------------------------------------------

def synthetic_bases(n_per_class=5, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for cls in ["layers ABCD", "rings", "smooth patches", "voronoi"]:
        for k in range(n_per_class):
            r, c = synthetic.tissue(rng)
            a = synthetic.adjacency(r, c)
            label = synthetic.arrange(cls, r, c, a, synthetic.BASE, rng)
            out[f"{cls} {k}"] = pd.DataFrame({"array_row": r - r.min(), "array_col": c - c.min(),
                                              "label": np.array(list("ABCD"))[label]})
    return out


def with_rare(sample, rng, share=0.02, minimum=40):
    """Sample with a rare cell type on a random scattering of spots; orig keeps the labels it replaced."""
    sample = sample.assign(label=sample["label"].astype(str), orig=sample["label"].astype(str))
    rare = rng.choice(len(sample), max(minimum, int(share * len(sample))), replace=False)
    label = sample["label"].to_numpy().copy()
    label[rare] = RARE
    return sample.assign(label=label)


def bases(dataset, seed=0):
    if dataset == "synthetic":
        out = synthetic_bases()
    elif dataset == "pig":
        spots = pd.read_csv(pig.DATA)
        out = {n: g[["array_row", "array_col", "label"]].reset_index(drop=True) for n, g in spots.groupby("sample_name")}
    else:
        samples, _ = load_dataset(dataset)
        out = {n: s.reset_index(drop=True) for n, s in samples.items()}
    rng = np.random.default_rng(seed)
    return {n: with_rare(s, rng) for n, s in out.items()}


# ---- noise and alterations ------------------------------------------------

def neighbors(adj, i):
    return adj.indices[adj.indptr[i]:adj.indptr[i + 1]]


def add_noise(sample, adj, kind, p, labels, rng):
    label = sample["label"].to_numpy().copy()
    if kind == "random":
        flip = np.flatnonzero(rng.random(len(label)) < p)
        for i in flip:
            label[i] = rng.choice([l for l in labels if l != label[i]])
    else:
        new = label.copy()
        for i in range(len(label)):
            nb = neighbors(adj, i)
            if len(nb) and (label[nb] != label[i]).any() and rng.random() < p:
                new[i] = label[rng.choice(nb)]
        label = new
    return sample.assign(label=label)


def patch(adj, center, k, rng):
    """The k spots nearest to center by steps on the grid, ties broken randomly."""
    seen, frontier = [center], [center]
    member = {center}
    while frontier and len(seen) < k:
        nxt = []
        for i in frontier:
            for j in rng.permutation(neighbors(adj, i)):
                if j not in member:
                    member.add(j)
                    nxt.append(j)
        rng.shuffle(nxt)
        seen.extend(nxt)
        frontier = nxt
    return np.array(seen[:k])


def alter(sample, adj, kind, k, rng):
    label = sample["label"].to_numpy().copy()
    common = label != RARE
    names, counts = np.unique(label[common], return_counts=True)
    if kind == "niche cluster":
        a, b = names[np.argsort(counts)[::-1][:2]]
        centers = [i for i in np.flatnonzero(label == a) if (label[neighbors(adj, i)] == b).any()]
        if not centers:
            centers = [i for i in np.flatnonzero(common) if (label[neighbors(adj, i)] != label[i]).any()]
        cluster = patch(adj, rng.choice(centers), k, rng)
        # Move scattered rare spots into the cluster, keeping the rare count unchanged
        new = cluster[label[cluster] != RARE]
        scattered = np.setdiff1d(np.flatnonzero(label == RARE), cluster)
        restore = rng.choice(scattered, min(len(new), len(scattered)), replace=False)
        label[restore] = sample["orig"].to_numpy()[restore]
        label[new] = RARE
        return sample.assign(label=label)
    else:
        # Contacts between labels, to find a label X that doesn't (or least) touch a common label Y
        codes = pd.Categorical(np.where(common, label, names[0]), categories=names).codes
        c = np.zeros((len(names), len(names)))
        for i in np.flatnonzero(common):
            nb = neighbors(adj, i)
            np.add.at(c[codes[i]], codes[nb[common[nb]]], 1)
        y = np.argmax(counts)
        touch = np.array([c[y, x] / counts[x] if x != y else np.inf for x in range(len(names))])
        x = np.argmin(touch)
        on_y = (codes == y) & common
        interior = [i for i in np.flatnonzero(on_y) if (label[neighbors(adj, i)] == names[y]).all()]
        centers = interior or list(np.flatnonzero(on_y))
        new_label = names[x]
    label[patch(adj, rng.choice(centers), k, rng)] = new_label
    return sample.assign(label=label)


# ---- distances ----------------------------------------------------------------

def contact_proportions(samples):
    labels = sorted(set().union(*(s["label"].astype(str) for s in samples)))
    out = []
    for s in samples:
        x, a = _graph(s, labels)
        c = (x.T @ a @ x).toarray().ravel()
        out.append(c / max(c.sum(), 1))
    return out


METHODS = {
    "composition only": lambda a, b: 1 - composition_similarity({"a": a, "b": b}).loc["a", "b"],
    # The package's euclidean rescales by the largest distance in the set, so
    # for a single pair it's 0 or 1; the unscaled distance is used instead
    "euclidean": lambda a, b: np.linalg.norm(np.subtract(*contact_proportions([a, b]))),
    "random_walk": lambda a, b: 1 - spatial_similarity({"a": a, "b": b}, metric="random_walk").loc["a", "b"],
    "npmi_heat t=4": lambda a, b: 1 - spatial_similarity({"a": a, "b": b}, metric="npmi_heat", t=4).loc["a", "b"],
    "moran": lambda a, b: 1 - spatial_similarity({"a": a, "b": b}, metric="moran").loc["a", "b"],
    "moran, exact tail": lambda a, b: 1 - spatial_similarity({"a": a, "b": b}, metric="moran", exact_tail=True).loc["a", "b"],
    "rw, unit diagonal": lambda a, b: 1 - spatial_similarity(
        {"a": a, "b": b}, metric="random_walk", unit_diagonal=True).loc["a", "b"],
    "rw, unit diag + log2": lambda a, b: 1 - spatial_similarity(
        {"a": a, "b": b}, metric="random_walk", unit_diagonal=True, transform="log2").loc["a", "b"],
}


def auc(pos, neg):
    """Probability a positive distance exceeds a negative one, counting ties as half."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return np.mean((pos > neg) + 0.5 * (pos == neg))


# ---- benchmark -------------------------------------------------------------------

def run(dataset, draws, seed=0):
    rng = np.random.default_rng(seed)
    dist = {}  # (noise, alteration or None, k, method) -> distances
    for base in bases(dataset).values():
        adj = _adjacency(base).tocsr()
        labels = sorted(base["label"].astype(str).unique())
        for noise, (kind, p) in NOISES.items():
            for _ in range(draws):
                a = add_noise(base, adj, kind, p, labels, rng)
                b = add_noise(base, adj, kind, p, labels, rng)
                pairs = {(None, 0): b}
                for alteration in ALTERATIONS:
                    for k in SIZES:
                        altered = alter(base, adj, alteration, k, rng)
                        pairs[(alteration, k)] = add_noise(altered, adj, kind, p, labels, rng)
                for (alteration, k), other in pairs.items():
                    for name, method in METHODS.items():
                        dist.setdefault((noise, alteration, k, name), []).append(method(a, other))
    return dist


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="*", default=["synthetic", "pig", "dlpfc", "heart"])
    parser.add_argument("--draws", type=int, default=3, help="noise draws per base section and noise level")
    args = parser.parse_args()
    for dataset in args.datasets:
        dist = run(dataset, args.draws)
        n = len(dist[(next(iter(NOISES)), None, 0, "random_walk")])
        print(f"\n===== {dataset}: AUC of positive vs negative pairs ({n} of each per cell)")
        cols = [(alt, k) for alt in ALTERATIONS for k in SIZES]
        for noise in NOISES:
            print(f"\n{noise}")
            print(f"{'method':18}" + "".join(f"{('niche' if alt == 'niche cluster' else 'domain') + f' k={k}':>13}" for alt, k in cols))
            for name in METHODS:
                neg = dist[(noise, None, 0, name)]
                print(f"{name:18}" + "".join(f"{auc(dist[(noise, alt, k, name)], neg):13.2f}" for alt, k in cols))
        print(flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
