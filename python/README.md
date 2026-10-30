<!-- README.md is generated from README.Rmd with knitr::knit("README.Rmd"). Please edit that file -->



<p align="center">
  <img src="https://raw.githubusercontent.com/george-hall-ucl/friendlyspots/main/assets/logo.gif" width="210" alt="friendlyspots logo">
</p>

Compute similarity between spatial transcriptomics samples based on which
labels (e.g. cell types) border one another.

## Installation

```bash
pip install "git+https://github.com/george-hall-ucl/friendlyspots#subdirectory=python"
```

Requires `pandas`, `numpy` and `scipy`. Also available as an
[R package](https://github.com/george-hall-ucl/friendlyspots/tree/main/R).

## Usage

`friendlyspots` is only compatible with original (non-HD) Visium.

```python
from friendlyspots import spatial_similarity

# Create one DataFrame per sample, with columns for array_row, array_col,
# label.
#   e.g. starting from a sample's AnnData object `adata`:
#   df1 = adata.obs[["array_row", "array_col"]].assign(
#       label=adata.obs["cell_type"]
#   )
samples = {"sample_1": df1, "sample_2": df2, "sample_3": df3}

sim = spatial_similarity(samples)
```

Returns a sample × sample DataFrame of similarity scores from 0 to 1, where 1
means spatially identical. With the default metric, 0 is the least similar pair
among the samples compared; other metrics give scores that don't depend on
which other samples are compared (see `metric` below).

Which metric to use: `metric="spectral"` did best in the
[benchmarks](#benchmarks), capturing structure beyond direct neighbors while
being robust to mislabelled spots, and `"random_walk"` is a simpler
alternative that also discounts how common each label is. The default,
`"euclidean"`, is kept for compatibility with earlier results, but mostly
reflects how much of each label samples have rather than how labels are
arranged.

### Options

- `normalize=False`: compare raw connection counts instead of proportions. This
  will make samples with different numbers of spots be considered spatially
  different even if their underlying spatial features are similar.
- `pools={"group_1": ["sample_1", "sample_2"], "group_2": ["sample_3"]}`: sum
  connection counts within groups before comparing.
- `smoothing=0.2`: first relabel isolated, likely mislabelled spots in each
  sample with `denoise_labels()` (below). The default relabels isolated spots
  and pairs; larger values also relabel larger groups. `smoothing=0` turns it
  off, e.g. to reproduce results computed without it, such as those of the pig
  esophagus paper (Durkin et al. 2026).
- `metric`: how connection counts are compared. All metrics other than the
  default give absolute scores, which don't depend on which other samples are
  compared, and require `normalize=True`.
  - `"euclidean"` (default): Euclidean distance between connection
    proportions, rescaled so the least similar pair scores 0.
  - `"random_walk"`: compares which labels each label borders, relative to its
    own total, so differences in how common each label is matter less. Simple,
    with no parameters. A label missing from one sample counts as a
    difference, in proportion to how common it is in the other.
  - `"npmi_heat"`: compares how much more or less often each pair of labels is
    near each other than their abundance predicts (normalized pointwise mutual
    information), removing the effect of composition. "Near" is a
    neighborhood of roughly `sqrt(3 * t)` spots (`t` defaults to 4), which
    makes it robust to scattered mislabelled spots. Pairs are weighted by how
    common the two labels are, so rare labels' noisy values count less. It
    compares the neighbors of labels present in both samples, including
    neighbors only one of them has.
  - `"spectral"`: averages `"random_walk"` with a comparison of the labels'
    graph Fourier spectra: how each label's spatial pattern divides between
    large domains and fine-grained mixing, and at which scales pairs of
    labels occupy the same or complementary places. Captures structure
    beyond direct neighbors, and is robust to scattered mislabelled spots.

### How the metrics work

All metrics start from each sample's **connection counts**. Every spot has up
to 6 neighbors on the Visium grid. For labels $i$ and $j$,

$$C_{ij} = \left|\lbrace (u, v) : u, v \text{ neighbors}, \ \text{label}(u) = i, \ \text{label}(v) = j \rbrace\right|,$$

so each touching pair of spots is counted once from each side and $C$ is
symmetric. The default `"euclidean"` metric divides $C$ by its total, so it
sums to 1, and takes the Euclidean distance between samples, divided by the
largest such distance among the samples compared.

These proportions mix two things: how much of each label a sample has
(**composition**) and how the labels are arranged. A sample with twice as many
spots of one label has more contacts involving that label even if the layout
is the same. The two other metrics separate these in different ways.

#### `"random_walk"`

Take a random spot of label $i$ and step to a random one of its neighbors. The
probability that the neighbor has label $j$ is

$$P_i(j) = \frac{\tilde C_{ij}}{r_i}, \qquad \tilde C = C + I, \qquad r_i = \sum_j \tilde C_{ij}.$$

In matrix form, $P = D^{-1} \tilde C$ with $D = \mathrm{diag}(r)$: the
transition matrix of a Markov chain on labels. Each row $P_i$ is a probability
distribution describing what label $i$ borders. Having more spots of label $i$
scales the whole row $C_{i\cdot}$, so $P_i$ doesn't change: the abundance of
label $i$ itself drops out.

The identity matrix $I$ adds one self loop per label. With thousands of
connections per sample it barely changes labels that are present, but it makes
every row a distribution: a label absent from a sample becomes a state the
walk never leaves, $P_i(i) = 1$, so it can be compared with the same label in
a sample that has it.

Two samples $a$ and $b$ are compared label by label, using the square root of
the Jensen-Shannon divergence with base-2 logarithms, which lies in $[0, 1]$:

$$D_i = \sqrt{\tfrac{1}{2} \mathrm{KL}\left(P_i^a  \parallel  M_i\right) + \tfrac{1}{2} \mathrm{KL}\left(P_i^b  \parallel  M_i\right)}, \qquad M_i = \tfrac{1}{2} \left(P_i^a + P_i^b\right).$$

Labels that are more common should count for more, so the label distances are
averaged with weights $\omega_i$, the label's share of connections averaged
over the two samples:

$$d(a, b) = \sum_i \omega_i D_i, \qquad \omega_i = \frac{1}{2} \left(\frac{r_i^a}{\sum_k r_k^a} + \frac{r_i^b}{\sum_k r_k^b}\right).$$

The weights sum to 1, and a label absent from one sample gets about half its
share in the other, so missing labels count in proportion to how common they
are where they appear. $\pi_i = r_i / \sum_k r_k$ is also a stationary
distribution of the chain ($\pi P = \pi$, as $\tilde C$ is symmetric): if
every label can reach every other, it's the long-run share of time the walk
spends on label $i$. The similarity is $1 - d(a, b)$.

#### `"npmi_heat"`

**1. Contacts over a neighborhood (heat diffusion).** Let $A$ be the
spot-by-spot neighbor matrix ($A_{uv} = 1$ if spots $u$ and $v$ touch), $L =
\mathrm{diag}(A \mathbf{1}) - A$ its graph Laplacian, and $X$ the
spot-by-label matrix with $X_{ui} = 1$ if spot $u$ has label $i$. Put one unit
of heat on every spot and let heat flow between touching spots for time $t$,
following $\frac{dh}{dt} = -L h$. Then

$$H = X^\top e^{-tL} X$$

has $H_{ij}$ equal to the heat starting on label-$i$ spots that ends up on
label-$j$ spots. Heat is conserved, so row $i$ sums to the number of label-$i$
spots. For small $t$, $H \approx X^\top X - t X^\top L X$, which only reflects
direct neighbors; larger $t$ reaches further, about $\sqrt{3t}$ spots. Averaging
over a neighborhood makes the result robust to scattered mislabelled spots.

**2. Removing composition (normalized pointwise mutual information).** Divide
$H$ by its total to get $P_{ij}$, with label shares $p_i = \sum_j P_{ij}$. If
labels were placed at random, labels $i$ and $j$ would be near each other in
proportion to their abundances, $P_{ij} \approx p_i p_j$. Pointwise mutual
information compares what's observed with that:

$$\mathrm{PMI}_{ij} = \log_2 \frac{P_{ij}}{p_i \, p_j},$$

which is 0 when $i$ and $j$ are near each other exactly as often as their
abundances predict, positive when more often and negative when less, whatever
the composition. It's then normalized to $[-1, 1]$:

$$\mathrm{NPMI}_{ij} = \frac{\mathrm{PMI}_{ij}}{-\log_2 P_{ij}},$$

with $\mathrm{NPMI}_{ij} = -1$ if $P_{ij} = 0$ (never near each other) and
$1$ if $P_{ij} = 1$.

**3. Comparing samples.** Rows are labels present in both samples ($R$);
columns are labels present in either ($K$), so a shared label bordering a
label the other sample lacks counts as a difference. NPMI of rare labels is
noisy, so each pair is weighted by the product of the two labels' shares
averaged over the samples, $\bar p_i = \frac{1}{2} (p_i^a + p_i^b)$:

$$d(a, b) = \frac{1}{2} \sqrt{\frac{\sum_{i \in R, j \in K} \bar p_i \bar p_j \left(\mathrm{NPMI}_{ij}^a - \mathrm{NPMI}_{ij}^b\right)^2}{\sum_{i \in R, j \in K} \bar p_i \bar p_j}}.$$

NPMI differences lie in $[-2, 2]$, so the factor $\frac{1}{2}$ keeps $d$ in
$[0, 1]$. The similarity is $1 - d(a, b)$, or 0 if $R$ is empty.

#### `"spectral"`

**1. Labels as signals on the spot graph.** For label $c$, let $y_c \in
\lbrace 0, 1 \rbrace^N$ indicate its spots. The normalized Laplacian
$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$ has eigendecomposition $\mathcal{L} = U
\Lambda U^\top$, with eigenvalues $0 = \lambda_1 \le \dots \le \lambda_N \le 2$.
The graph Fourier transform $\hat y_c = U^\top y_c$ writes $y_c$ as a sum of the
eigenvectors $u_i$, and an eigenvector varies across the tissue more the
larger its eigenvalue, since

$$u_i^\top \mathcal{L} \, u_i = \sum_{p, q \text{ neighbors}} \left(\frac{u_{i,p}}{\sqrt{d_p}} - \frac{u_{i,q}}{\sqrt{d_q}}\right)^2 = \lambda_i,$$

with $d_p$ the number of neighbors of spot $p$. Small $\lambda_i$ are smooth
patterns spanning large domains; large $\lambda_i$ change from spot to spot,
like salt-and-pepper mixing. The zero-frequency part ($\lambda = 0$, one
eigenvector proportional to $\sqrt{d_p}$ per connected piece of tissue) only
reflects how much of each label there is, so it's removed.

**2. Energy per frequency band.** The energy that labels $c$ and $c'$ share in
band $b$ is

$$E_b(c, c') = \sum_{i : \lambda_i \in b} \hat y_{c,i} \, \hat y_{c',i} = y_c^\top P_b \, y_{c'},$$

where $P_b$ projects onto the eigenvectors in band $b$. Individual eigenvectors
are only defined up to sign, and up to rotation when eigenvalues repeat, but
$P_b$ isn't, so $E_b$ is alignment-free. The bands are log-spaced in $\lambda
N$, from 1 to $10^4$: the eigenvalues of patterns covering a given share of the
tissue scale with $1 / N$, so this puts them in the same band whatever the size
of the section.

**3. Comparing samples.** Each label's spectrum shape is its energy per band
divided by its total, $s_c(b) = E_b(c, c) / \sum_{b'} E_{b'}(c, c)$, compared
between samples with $\sqrt{\mathrm{JS}}$ as for `"random_walk"`. Pairs of
labels are compared by their coherence,

$$\kappa_{cc'}(b) = \frac{E_b(c, c')}{\sqrt{E_b(c, c) \, E_b(c', c')}} \in [-1, 1],$$

which is near $-1$ in bands where the two labels occupy complementary places,
near 0 where they're unrelated and near 1 where they're found together. With
label shares $\omega_c$ averaged over the two samples,

$$d_\text{spectral}(a, b) = \frac{1}{2} \left( \frac{\sum_c \omega_c \sqrt{\mathrm{JS}}\left(s_c^a, s_c^b\right)}{\sum_c \omega_c} + \frac{\sum_{c < c'} \omega_c \omega_{c'} \, \overline{\left|\kappa_{cc'}^a - \kappa_{cc'}^b\right|} / 2}{\sum_{c < c'} \omega_c \omega_{c'}} \right),$$

where the bar is the mean over bands, a label present in only one sample has
distance 1, and pairs are compared only if both labels are in both samples.
The metric's distance is $\frac{1}{2} \left(d_\text{random walk} +
d_\text{spectral}\right)$, and the similarity is 1 minus it.

**4. Without an eigendecomposition.** Computing $U$ takes time proportional to
$N^3$, so the band energies are instead estimated with the kernel polynomial
method. $M = \mathcal{L} - I$ has eigenvalues in $[-1, 1]$, where the Chebyshev
polynomials $T_0(M) = I$, $T_1(M) = M$, $T_{k+1}(M) = 2 M \, T_k(M) - T_{k-1}(M)$
are defined, so the moments $\mu_k = y_c^\top T_k(M) \, y_{c'}$ only need
sparse matrix-vector products. Writing $\lambda - 1 = \cos \theta$, the energy
in a band from $\lambda_\text{lo}$ to $\lambda_\text{hi}$ is then

$$E \approx g_0 \mu_0 \frac{\theta_\text{lo} - \theta_\text{hi}}{\pi} + \sum_{k=1}^{K-1} 2 g_k \mu_k \frac{\sin k \theta_\text{lo} - \sin k \theta_\text{hi}}{k \pi},$$

where the Jackson weights $g_k$ damp the ringing of a truncated series. With
$K = 1000$ moments the total energy is exact, and frequencies are resolved to
about $(\pi / K)^2$ near $\lambda = 0$, finer than the gaps between a section's
smallest eigenvalues.

### Label smoothing

`denoise_labels(sample, lam=0.2)` relabels isolated spots that are likely
mislabelled, and `spatial_similarity()` applies it to each sample with
`lam=smoothing`. It can also be used on its own, e.g. to check which spots it
would change:

```python
from friendlyspots import denoise_labels

smoothed = denoise_labels(df1, lam=0.2)
changed = df1[df1["label"] != smoothed["label"]]
```

Given a sample's labels $y$, it finds new labels $x$ minimizing

$$E(x) = \sum_p [x_p \neq y_p] + \lambda \sum_{p, q \text{ neighbors}} [x_p \neq x_q],$$

where $[\cdot]$ is 1 if true and 0 otherwise. The first term counts changed
labels and the second the edges between neighbors with different labels, the
total length of boundaries between labels, which is half the off-diagonal sum
of $C$. A group of $k$ spots with $P$ edges to other labels is therefore
relabelled when $\lambda P > k$. A spot has at most 6 neighbors, so values of
$\lambda$ below $1/6$ change nothing, $0.2$ relabels isolated spots and pairs,
and larger values relabel larger groups and smooth boundaries, also erasing
labels that are genuinely scattered. $E$ is minimized with alpha-expansion
graph cuts (Boykov et al. 2001): for each label in turn, a maximum flow decides
which spots switch to it, repeated until no spot changes.

### Benchmarks

How well each metric retrieves samples known to belong together, as mean
average precision (mAP: 1 means every sample ranks all samples of its group
above all others). Composition only is a baseline, not a metric option: it
compares label proportions, ignoring where spots are. Metrics use the
default `smoothing=0.2`. Best per row in bold.
Scripts and data downloads to reproduce these are in
[`benchmarks/`](https://github.com/george-hall-ucl/friendlyspots/tree/main/python/benchmarks).

| Dataset | Task | Chance | Composition only | `euclidean` | `random_walk` | `npmi_heat` t=4 | `npmi_heat` t=16 | `spectral` |
|---|---|---|---|---|---|---|---|---|
| Synthetic | Arrangement, fixed composition | 0.2 | 0.25 | 0.88 | 0.85 | 0.86 | 0.78 | **0.91** |
| Synthetic | Arrangement, varied composition | 0.2 | 0.22 | 0.34 | 0.72 | 0.75 | 0.67 | **0.76** |
| Synthetic | ... + boundary label noise (30%) | 0.2 | 0.23 | 0.36 | 0.70 | 0.74 | 0.68 | **0.77** |
| Synthetic | ... + random label noise (15%) | 0.2 | 0.23 | 0.38 | 0.54 | **0.66** | 0.65 | 0.63 |
| Pig esophagus | Time point | 0.40 | 0.61 | 0.59 | 0.57 | 0.63 | 0.62 | **0.64** |
| Human cortex | Donor | 0.42 | 0.96 | 0.93 | **1.00** | **1.00** | **1.00** | **1.00** |
| Human cortex | Adjacent section | 0.28 | 0.85 | 0.86 | 0.86 | 0.86 | 0.86 | **0.94** |
| Human cortex | Donor, random label noise (15%) | 0.42 | 0.95 | 0.94 | **1.00** | **1.00** | **1.00** | **1.00** |
| Human cortex | Adjacent section, random label noise (15%) | 0.28 | 0.82 | 0.84 | **0.90** | 0.88 | 0.84 | **0.90** |
| Human cortex | Donor, random label noise (30%) | 0.42 | 0.96 | 0.94 | 0.97 | 0.99 | **1.00** | **1.00** |
| Human cortex | Adjacent section, random label noise (30%) | 0.28 | 0.85 | 0.85 | 0.88 | 0.87 | 0.85 | **0.93** |
| Human cortex | Donor, random crops | 0.37 | 0.61 | 0.60 | **0.72** | 0.70 | 0.67 | 0.68 |
| Human cortex | Section, random crops | 0.13 | 0.29 | 0.30 | **0.38** | 0.33 | 0.31 | 0.33 |
| Human heart | Patient group | 0.46 | 0.79 | 0.77 | 0.82 | 0.85 | 0.72 | **0.87** |

Synthetic scores are means over 10 random datasets (standard deviations
0.03-0.08); noise and crop scores are means over 5 random draws.

- **Synthetic**: generated tissue sections of random size, shape and
  orientation with 4 labels, in 6 arrangements (5 sections each): two layer
  orders, rings, smooth patches, randomly assigned patches and fully mixed.
  Composition is either the same for all sections or varies between them, so
  arrangement can be tested separately from composition.
- **Pig esophagus**: the 10 sections of the package's test data (Durkin et
  al. 2026), labelled by cell type: pre-implant, 1, 3 and 6 months after
  implantation, and native tissue. The task is to retrieve sections of the
  same time point.
- **Human cortex**: 12 sections of dorsolateral prefrontal cortex (Maynard et
  al. 2021) with manual annotations of cortical layers, from 3 donors, cut as
  pairs of adjacent sections. Also tested with random label noise, and with
  random crops of ~40% of each section, whose composition depends on where
  they fall while the layer order stays the same.
- **Human heart**: 15 sections after myocardial infarction and from controls
  (Kuppe et al. 2022), labelled by each spot's dominant cell type. The task is
  to retrieve sections of the same patient group: myogenic (control and
  remote zone), ischaemic or fibrotic.

### Example

Uses four mouse brain sections from 10x Genomics (two anterior, two posterior),
with each spot labelled by its most enriched of five marker genes.

<details><summary>Setup: download and label the data, define plotting helpers (click to expand)</summary>


``` python
# To rebuild this README from README.Rmd, run knitr::knit("README.Rmd") in R
# from the python/ directory, with RETICULATE_PYTHON set to a Python that has
# friendlyspots installed (rmarkdown::render breaks the <details> block).
# Requires squidpy, matplotlib and seaborn.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import squidpy as sq

markers = ["Mbp", "Slc17a7", "Penk", "Hpca", "Gabra6"]
ids = {
    "anterior_1": "V1_Mouse_Brain_Sagittal_Anterior",
    "anterior_2": "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
    "posterior_1": "V1_Mouse_Brain_Sagittal_Posterior",
    "posterior_2": "V1_Mouse_Brain_Sagittal_Posterior_Section_2",
}
cache = Path.home() / ".cache" / "friendlyspots"

adatas = {}
labels = {}
for name, sample_id in ids.items():
    adata = sq.datasets.visium(sample_id, base_dir=cache)
    adata.var_names_make_unique()
    counts = adata[:, markers].X.toarray()
    top = (counts / counts.mean(axis=0)).argmax(axis=1)
    adatas[name] = adata
    labels[name] = np.array(markers)[top]


def plot_slices(samples):
    palette = ["#CC79A7", "#D55E00", "#0072B2", "#F0E442", "#009E73"]
    colors = dict(zip(sorted(markers), palette))
    fig, axes = plt.subplots(1, len(samples), layout="constrained")
    for ax, (name, df) in zip(axes, samples.items()):
        ax.scatter(
            df["array_col"],
            -df["array_row"] * np.sqrt(3),
            c=df["label"].map(colors),
            s=3,
            linewidths=0,
        )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for label, color in colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(colors),
        frameon=False,
    )
    plt.show()


def plot_similarity(sim):
    style = {"font.size": 14, "font.weight": "bold", "figure.autolayout": True}
    with plt.rc_context(style):
        ax = sns.heatmap(
            sim,
            mask=np.tril(np.ones(sim.shape, dtype=bool), k=-1),
            annot=True,
            fmt=".2f",
            cmap="inferno",
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={"label": "Similarity"},
        )
        ax.invert_yaxis()
        plt.xticks(rotation=45, ha="right")
    plt.show()
```

</details>

Each sample is a DataFrame of spot array coordinates and labels:


``` python
from friendlyspots import spatial_similarity

samples = {
    name: adata.obs[["array_row", "array_col"]].assign(label=labels[name])
    for name, adata in adatas.items()
}
print(samples["anterior_1"].head())
#>                     array_row  array_col    label
#> AAACAAGTATCTCCCA-1         50        102      Mbp
#> AAACACCAATAACTGC-1         59         19  Slc17a7
#> AAACAGAGCGACTCCT-1         14         94  Slc17a7
#> AAACAGCTTTCAGAAG-1         43          9      Mbp
#> AAACAGGGTCTATATT-1         47         13     Penk
```


``` python
plot_slices(samples)
```

<img src="https://raw.githubusercontent.com/george-hall-ucl/friendlyspots/main/python/figures/README-slices-1.png" alt="" width="75%" style="display: block; margin: auto;" />


``` python
sim = spatial_similarity(samples)
print(sim.round(2))
#>              anterior_1  anterior_2  posterior_1  posterior_2
#> anterior_1         1.00        0.93         0.00         0.01
#> anterior_2         0.93        1.00         0.03         0.04
#> posterior_1        0.00        0.03         1.00         0.92
#> posterior_2        0.01        0.04         0.92         1.00
plot_similarity(sim)
```

<img src="https://raw.githubusercontent.com/george-hall-ucl/friendlyspots/main/python/figures/README-similarity-3.png" alt="" width="50%" style="display: block; margin: auto;" />

## Citation

The algorithm and original implementation of `friendlyspots` are based on the
methodology in:

- Durkin _et al_. "Functional integration of an autologous engineered esophagus
  in a large-animal model." _Nature Biotechnology_ (2026).
- Budinger _et al_. "An in vivo and in vitro spatiotemporal profile of human
  midbrain development." _Nature Communications_ (2026).

## Licence

Copyright (C) 2026 University College London. Licensed under GPL-3.0-or-later.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.
