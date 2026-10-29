import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from friendlyspots import spatial_similarity
from friendlyspots.similarity import _npmi_distance

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


@pytest.mark.parametrize("metric", ["random_walk", "npmi_heat"])
def test_requires_normalize(metric):
    with pytest.raises(ValueError, match="requires normalize=True"):
        spatial_similarity(SAMPLES, normalize=False, metric=metric)


def test_unknown_metric():
    with pytest.raises(ValueError, match="metric must be"):
        spatial_similarity(SAMPLES, metric="kl")


def test_random_walk():
    """Random walk compares each shared label's neighbor-label distribution, weighted by label share."""
    # a's neighbors: A [1/2, 1/2], B [2/3, 1/3]; b's neighbors: A and B [1, 0], C [0, 1]
    # Label shares: A a 4/6, b 2/6; B a 3/4, b 1/4; C shares only b with A and B
    js_a = (np.log2(6 / 7) + np.log2(6 / 5)) / 4 + (2 * np.log2(8 / 7) + np.log2(4 / 5)) / 6
    sim = spatial_similarity(SAMPLES, metric="random_walk")
    assert_sim(sim, 1 - (17 / 24) * np.sqrt(js_a), 0, 0)


def test_npmi_distance():
    """NPMI distance compares how much more or less often labels touch than chance, weighted by label shares."""
    # Connection counts of A, B and C (labels a, b)
    a, b, c = np.array([[2, 2], [2, 0]]), np.array([[2, 1], [1, 0]]), np.array([[0, 0], [0, 2]])
    # A: P = [[1/3, 1/3], [1/3, 0]]; B: P = [[1/2, 1/4], [1/4, 0]]; b-b never touch (-1) in both
    npmi_ab_a, npmi_ab_b = np.log2(3 / 2) / np.log2(3), np.log2(4 / 3) / 2
    diff_aa = np.log2(3 / 4) / np.log2(3) - np.log2(8 / 9)
    # Mean label shares: A, B [17/24, 7/24]; pair weights are their products
    ab = np.sqrt((17 / 24) ** 2 * diff_aa**2 + 2 * (17 * 7 / 24**2) * (npmi_ab_a - npmi_ab_b) ** 2) / 2
    # C has only b, touching itself (NPMI 1; -1 in A and B) and never a (-1),
    # so only b's row is compared, over both labels. Mean shares: A, C [1/3, 2/3]; B, C [3/8, 5/8]
    ac = np.sqrt((2 / 9 * (npmi_ab_a + 1) ** 2 + 4 / 9 * 4) / (6 / 9)) / 2
    bc = np.sqrt((15 / 64 * (npmi_ab_b + 1) ** 2 + 25 / 64 * 4) / (40 / 64)) / 2
    np.testing.assert_allclose([_npmi_distance(a, b), _npmi_distance(a, c), _npmi_distance(b, c)], [ab, ac, bc])


def test_npmi_heat_sees_beyond_neighbors():
    """npmi_heat tells apart samples whose direct-neighbor contacts are identical."""
    # Both have contacts a-a, b-b and a-b once each and half a, half b
    # labels, but only in "joined" are the a-a and b-b pairs near each other
    joined = pd.DataFrame({"array_row": [0, 0, 0, 0], "array_col": [0, 2, 4, 6], "label": ["a", "a", "b", "b"]})
    apart = pd.DataFrame({"array_row": [0] * 6, "array_col": [0, 2, 10, 12, 20, 22], "label": ["a", "a", "b", "b", "a", "b"]})
    pair = {"joined": joined, "apart": apart}
    assert spatial_similarity(pair, metric="random_walk").loc["joined", "apart"] == 1
    assert spatial_similarity(pair, metric="npmi_heat").loc["joined", "apart"] < 1


def test_npmi_heat_t():
    """Larger t diffuses further, so changes the result; t must be positive and is only for npmi_heat."""
    near = spatial_similarity(SAMPLES, metric="npmi_heat", t=0.5)
    far = spatial_similarity(SAMPLES, metric="npmi_heat", t=8)
    assert not np.allclose(near.to_numpy(), far.to_numpy())
    pd.testing.assert_frame_equal(spatial_similarity(SAMPLES, metric="npmi_heat"),
                                  spatial_similarity(SAMPLES, metric="npmi_heat", t=4))
    with pytest.raises(ValueError, match="t must be positive"):
        spatial_similarity(SAMPLES, metric="npmi_heat", t=0)
    with pytest.raises(ValueError, match="only used with metric='npmi_heat'"):
        spatial_similarity(SAMPLES, metric="random_walk", t=4)


@pytest.mark.parametrize("metric", ["random_walk", "npmi_heat"])
def test_composition_metrics_identical_samples(metric):
    """Identical samples are fully similar."""
    sim = spatial_similarity({"A": SAMPLES["A"], "B": SAMPLES["A"]}, metric=metric)
    np.testing.assert_allclose(sim.to_numpy(), np.ones((2, 2)))


@pytest.mark.parametrize("metric", ["random_walk", "npmi_heat"])
def test_composition_metrics_no_connections(metric):
    """Samples with no neighboring spots yield NA."""
    sim = spatial_similarity(SAMPLES, metric=metric)
    d = pd.DataFrame({"array_row": [0, 0], "array_col": [0, 4], "label": ["a", "a"]})
    with pytest.warns(UserWarning, match="no neighboring spots: D"):
        with_d = spatial_similarity({**SAMPLES, "D": d}, metric=metric)
    pd.testing.assert_frame_equal(with_d.loc[["A", "B", "C"], ["A", "B", "C"]], sim)
    assert with_d["D"].isna().all() and with_d.loc["D"].isna().all()


@pytest.mark.parametrize("metric", ["random_walk", "npmi_heat"])
def test_composition_metrics_pools(metric):
    """Single-sample pools match the samples; pooling sums before comparing."""
    sim = spatial_similarity(SAMPLES, metric=metric)
    single = spatial_similarity(SAMPLES, metric=metric, pools={"A": ["A"], "B": ["B"], "C": ["C"]})
    pd.testing.assert_frame_equal(single, sim)
    pooled = spatial_similarity(SAMPLES, metric=metric, pools={"AB": ["A", "B"], "A": ["A"]})
    assert 0 < pooled.loc["AB", "A"] < 1


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
