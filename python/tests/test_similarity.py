import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from friendlyspots import spatial_similarity
from friendlyspots.similarity import _prepare_counts

SAMPLES = {
    "A": pd.DataFrame({"array_row": [0, 0, 1], "array_col": [0, 2, 1], "label": ["a", "b", "a"]}),
    "B": pd.DataFrame({"array_row": [0, 0, 0, 2], "array_col": [0, 2, 4, 0], "label": ["a", "a", "b", "b"]}),
    "C": pd.DataFrame({"array_row": [0, 0], "array_col": [0, 2], "label": ["b", "b"]}),
}


def assert_sim(sim, ab, ac, bc):
    expected = np.array([[1, ab, ac], [ab, 1, bc], [ac, bc, 1]])
    np.testing.assert_allclose(sim.to_numpy(), expected)
    assert list(sim.columns) == list(sim.index)


def test_neighbor_counts():
    """Hexagonal neighbor counts give similarity 1 - d / max(d)."""
    sim = spatial_similarity(SAMPLES, normalize=False)
    assert_sim(sim, 1 - np.sqrt(2) / 4, 0, 1 - np.sqrt(10) / 4)


def test_names_in_input_order():
    """Output row/column names respect input sample order."""
    sim = spatial_similarity(dict(reversed(SAMPLES.items())), normalize=False)
    assert list(sim.index) == ["C", "B", "A"]


def test_identical_samples():
    """Identical samples are seen as fully similar."""
    sim = spatial_similarity({"A": SAMPLES["A"], "B": SAMPLES["A"]})
    np.testing.assert_array_equal(sim.to_numpy(), np.ones((2, 2)))


def test_categorical_labels():
    """Categorical labels yield the same output as string labels."""
    sim = spatial_similarity(SAMPLES, normalize=False)
    categories = {n: s.assign(label=s["label"].astype("category")) for n, s in SAMPLES.items()}
    pd.testing.assert_frame_equal(spatial_similarity(categories, normalize=False), sim)


def test_missing_labels():
    """Labels read as missing (e.g. NA, nan, None) become one label called 'NA', with a warning."""
    sim = spatial_similarity(SAMPLES, normalize=False)
    # pd.read_csv() turns labels written as NA, nan, None, etc. into NaN
    csv = pd.read_csv(io.StringIO(
        "sample,array_row,array_col,label\n"
        "A,0,0,a\nA,0,2,NA\nA,1,1,a\n"
        "B,0,0,a\nB,0,2,a\nB,0,4,nan\nB,2,0,nan\n"
        "C,0,0,None\nC,0,2,None\n"
    ))
    from_file = {name: s.drop(columns="sample") for name, s in csv.groupby("sample")}
    with pytest.warns(UserWarning, match="treated as a label called 'NA'"):
        pd.testing.assert_frame_equal(spatial_similarity(from_file, normalize=False), sim)


def test_normalize():
    """Normalize divides counts by their total."""
    dmax = np.sqrt(11 / 8)
    sim = spatial_similarity(SAMPLES)
    assert_sim(sim, 1 - (np.sqrt(6) / 12) / dmax, 1 - np.sqrt(4 / 3) / dmax, 0)


def test_no_connections_with_normalize():
    """Samples where no spot has any neighbor yield NA and a warning."""
    sim = spatial_similarity(SAMPLES)
    # D's two spots don't touch, so it has no connections to normalize
    d = pd.DataFrame({"array_row": [0, 0], "array_col": [0, 4], "label": ["a", "a"]})
    with pytest.warns(UserWarning, match="no neighboring spots: D"):
        with_d = spatial_similarity({**SAMPLES, "D": d})
    pd.testing.assert_frame_equal(with_d.loc[["A", "B", "C"], ["A", "B", "C"]], sim)
    assert with_d["D"].isna().all() and with_d.loc["D"].isna().all()


def test_pools():
    """Pools sum counts of their samples."""
    pools = {"p1": ["A", "B"], "p2": ["C"], "p3": ["A"]}
    sim = spatial_similarity(SAMPLES, normalize=False, pools=pools)
    assert list(sim.index) == ["p1", "p2", "p3"]
    assert_sim(sim, 0, 1 - np.sqrt(6) / np.sqrt(38), 1 - 4 / np.sqrt(38))


@pytest.mark.parametrize("metric", ["random_walk"])
def test_requires_normalize(metric):
    with pytest.raises(ValueError, match="requires normalize=True"):
        spatial_similarity(SAMPLES, normalize=False, metric=metric)


def test_unknown_metric():
    with pytest.raises(ValueError, match="metric must be"):
        spatial_similarity(SAMPLES, metric="kl")


def sqrt_js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    kl = lambda x: sum(xi * np.log2(xi / mi) for xi, mi in zip(x, m) if xi > 0)
    return np.sqrt((kl(p) + kl(q)) / 2)


def test_random_walk():
    """Random walk compares each label's next-step distribution, weighted by label share."""
    # Connection counts plus one self loop per label, rows [a, b]:
    # A [[3, 2], [2, 1]], B [[3, 1], [1, 1]], C [[1, 0], [0, 3]] (C has no a, so a always stays)
    # Label shares (row sums / total): A [5/8, 3/8], B [2/3, 1/3], C [1/4, 3/4]
    ab = 31 / 48 * sqrt_js([3 / 5, 2 / 5], [3 / 4, 1 / 4]) + 17 / 48 * sqrt_js([2 / 3, 1 / 3], [1 / 2, 1 / 2])
    ac = 7 / 16 * sqrt_js([3 / 5, 2 / 5], [1, 0]) + 9 / 16 * sqrt_js([2 / 3, 1 / 3], [0, 1])
    bc = 11 / 24 * sqrt_js([3 / 4, 1 / 4], [1, 0]) + 13 / 24 * sqrt_js([1 / 2, 1 / 2], [0, 1])
    sim = spatial_similarity(SAMPLES, metric="random_walk")
    assert_sim(sim, 1 - ab, 1 - ac, 1 - bc)


@pytest.mark.parametrize("metric", ["random_walk"])
def test_composition_metrics_identical_samples(metric):
    """Identical samples are fully similar."""
    sim = spatial_similarity({"A": SAMPLES["A"], "B": SAMPLES["A"]}, metric=metric)
    np.testing.assert_allclose(sim.to_numpy(), np.ones((2, 2)))


@pytest.mark.parametrize("metric", ["random_walk"])
def test_composition_metrics_no_connections(metric):
    """Samples with no neighboring spots yield NA."""
    sim = spatial_similarity(SAMPLES, metric=metric)
    d = pd.DataFrame({"array_row": [0, 0], "array_col": [0, 4], "label": ["a", "a"]})
    with pytest.warns(UserWarning, match="no neighboring spots: D"):
        with_d = spatial_similarity({**SAMPLES, "D": d}, metric=metric)
    pd.testing.assert_frame_equal(with_d.loc[["A", "B", "C"], ["A", "B", "C"]], sim)
    assert with_d["D"].isna().all() and with_d.loc["D"].isna().all()


@pytest.mark.parametrize("metric", ["random_walk"])
def test_composition_metrics_pools(metric):
    """Single-sample pools match the samples; pooling sums before comparing."""
    sim = spatial_similarity(SAMPLES, metric=metric)
    single = spatial_similarity(SAMPLES, metric=metric, pools={"A": ["A"], "B": ["B"], "C": ["C"]})
    pd.testing.assert_frame_equal(single, sim)
    pooled = spatial_similarity(SAMPLES, metric=metric, pools={"AB": ["A", "B"], "A": ["A"]})
    assert 0 < pooled.loc["AB", "A"] < 1


def hexagon(radius=5):
    """Spots within radius steps of the center, labelled by sector and distance, so no symmetry maps it to itself."""
    rows, cols, labels = [], [], []
    for r in range(-radius, radius + 1):
        for c in range(-2 * radius, 2 * radius + 1):
            # Doubled-width coordinates to cube coordinates, for the hexagonal distance
            q = (c - r) // 2
            if (r + c) % 2 or max(abs(q), abs(r), abs(q + r)) > radius:
                continue
            angle = np.arctan2(r * np.sqrt(3) / 2, c / 2)
            label = "abc"[int((angle + np.pi) / (2 * np.pi) * 3) % 3] if abs(q) + abs(r) + abs(q + r) > 4 else "d"
            rows.append(r + radius); cols.append(c + 2 * radius); labels.append(label)
    return pd.DataFrame({"array_row": rows, "array_col": cols, "label": labels})


def rotate60(sample):
    r = sample["array_row"].to_numpy(); q = (sample["array_col"].to_numpy() - r) // 2
    x, z = -r, -(-q - r)  # cube (q, -q-r, r) rotated by 60 degrees
    rows, cols = z, 2 * x + z
    return sample.assign(array_row=rows - rows.min(), array_col=cols - cols.min() + ((rows - rows.min()) % 2 != (cols - cols.min()) % 2))


@pytest.mark.parametrize("metric, options", [
    ("euclidean", {}), ("random_walk", {}),
    ("euclidean", {"unit_diagonal": True}), ("random_walk", {"unit_diagonal": True}),
])
def test_alignment_free(metric, options):
    """Rotating, mirroring or reordering a sample's spots on the grid doesn't change its similarities."""
    s = hexagon()
    variants = {"rotated": rotate60(s), "rotated twice": rotate60(rotate60(s)),
                "mirrored": s.assign(array_col=s["array_col"].max() - s["array_col"]), "shuffled": s.sample(frac=1, random_state=0)}
    for name, v in variants.items():
        assert len(set(zip(v["array_row"], v["array_col"]))) == len(s)
        sim = spatial_similarity({"original": s, name: v}, metric=metric, **options)
        np.testing.assert_allclose(sim.to_numpy(), 1, atol=1e-9, err_msg=name)


def test_prepare_counts():
    """Self loops and the unit diagonal give the expected matrices."""
    c = np.array([[2.0, 2.0], [2.0, 0.0]])  # sample A
    np.testing.assert_allclose(_prepare_counts(c, False, False), c)
    np.testing.assert_allclose(_prepare_counts(c, True, False), [[3, 2], [2, 1]])
    unit = [[1, 2 / np.sqrt(3)], [2 / np.sqrt(3), 1]]
    np.testing.assert_allclose(_prepare_counts(c, False, True), unit)
    np.testing.assert_allclose(_prepare_counts(c, True, True), unit)  # self loops aren't added twice


def test_random_walk_unit_diagonal():
    """With unit_diagonal, random_walk compares rows of C_ij / sqrt(C_ii C_jj), with self loops."""
    # A: C + I = [[3, 2], [2, 1]], unit diagonal [[1, 2/sqrt 3], [2/sqrt 3, 1]]
    # B: C + I = [[3, 1], [1, 1]], unit diagonal [[1, 1/sqrt 3], [1/sqrt 3, 1]]
    # Both labels' rows have the same total, so each is weighted 1/2
    ua, ub = 2 / np.sqrt(3), 1 / np.sqrt(3)
    pa, pb = np.array([1, ua]) / (1 + ua), np.array([1, ub]) / (1 + ub)
    expected = (sqrt_js(pa, pb) + sqrt_js(pa[::-1], pb[::-1])) / 2
    sim = spatial_similarity({"A": SAMPLES["A"], "B": SAMPLES["B"]}, metric="random_walk", unit_diagonal=True)
    np.testing.assert_allclose(sim.loc["A", "B"], 1 - expected)


def test_pig_esophagus_paper():
    """Reproduces the pig esophagus paper's similarities."""
    # Durkin et al. (2026)
    spots = pd.read_csv(Path(__file__).parent / "pig_esophagus_spots.csv.xz")
    samples = {name: s for name, s in spots.groupby("sample_name")}
    pools = {
        "Pre-implant": ["Pre-implant"],
        "1 month": ["1 month (D)"],
        "3 months": ["3 months (E)", "3 months (F)"],
        "6 months": ["6 months (B)", "6 months (C)"],
        "Native": ["Native DP1", "Native DP2", "Native DP3", "Native (C)"],
    }
    sim = spatial_similarity(samples, pools=pools)
    paper = np.array([
        [1.00, 0.01, 0.03, 0.09, 0.00],
        [0.01, 1.00, 0.90, 0.68, 0.27],
        [0.03, 0.90, 1.00, 0.70, 0.28],
        [0.09, 0.68, 0.70, 1.00, 0.52],
        [0.00, 0.27, 0.28, 0.52, 1.00],
    ])
    # The paper's 7.5 px neighbor cutoff missed a few neighbors in Native DP2
    paper[0, 1] = paper[1, 0] = 0.02
    assert list(sim.index) == list(pools)
    np.testing.assert_allclose(sim.round(2).to_numpy(), paper, atol=1e-12)
