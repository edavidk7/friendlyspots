<!-- README.md is generated from README.Rmd with knitr::knit("README.Rmd"). Please edit that file -->



<p align="center">
  <img src="https://raw.githubusercontent.com/edavidk7/friendlyspots/metrics-and-benchmarks/assets/logo.gif" width="210" alt="friendlyspots logo">
</p>

Compute similarity between spatial transcriptomics samples based on which
labels (e.g. cell types) border one another.

## Installation

```bash
pip install "git+https://github.com/edavidk7/friendlyspots@metrics-and-benchmarks#subdirectory=python"
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

Which metric to use: `metric="npmi_heat"` did best in most of the
[benchmarks](#benchmarks), removing the effect of composition while being
robust to scattered mislabelled spots, and `"random_walk"` is a simpler
alternative with no parameters that also discounts how common each label is.
The default, `"euclidean"`, is kept for compatibility with earlier results, but
mostly reflects how much of each label samples have rather than how labels are
arranged.

### Options

- `normalize=False`: compare raw connection counts instead of proportions. This
  will make samples with different numbers of spots be considered spatially
  different even if their underlying spatial features are similar.
- `pools={"group_1": ["sample_1", "sample_2"], "group_2": ["sample_3"]}`: sum
  connection counts within groups before comparing.
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

### Benchmarks

How well each metric retrieves samples known to belong together, as mean
average precision (mAP: 1 means every sample ranks all samples of its group
above all others). Composition only is a baseline, not a metric option: it
compares label proportions, ignoring where spots are. Best per row in bold.
Scripts and data downloads to reproduce these are in
[`benchmarks/`](https://github.com/edavidk7/friendlyspots/tree/metrics-and-benchmarks/python/benchmarks).

| Dataset | Task | Chance | Composition only | `euclidean` | `random_walk` | `npmi_heat` t=4 | `npmi_heat` t=16 |
|---|---|---|---|---|---|---|---|
| Synthetic | Arrangement, fixed composition | 0.2 | 0.25 | **0.89** | 0.85 | 0.86 | 0.79 |
| Synthetic | Arrangement, varied composition | 0.2 | 0.22 | 0.37 | 0.72 | **0.75** | 0.67 |
| Synthetic | ... + boundary label noise (30%) | 0.2 | 0.23 | 0.37 | 0.71 | **0.74** | 0.68 |
| Synthetic | ... + random label noise (15%) | 0.2 | 0.23 | 0.37 | 0.44 | 0.64 | **0.65** |
| Pig esophagus | Time point | 0.40 | 0.61 | 0.59 | 0.59 | **0.63** | 0.61 |
| Human cortex | Donor | 0.42 | 0.96 | 0.93 | **1.00** | **1.00** | **1.00** |
| Human cortex | Adjacent section | 0.28 | 0.85 | **0.86** | **0.86** | **0.86** | **0.86** |
| Human cortex | Donor, random label noise (15%) | 0.42 | 0.95 | 0.94 | 0.97 | **1.00** | **1.00** |
| Human cortex | Adjacent section, random label noise (15%) | 0.28 | 0.82 | 0.84 | **0.89** | **0.89** | 0.85 |
| Human cortex | Donor, random label noise (30%) | 0.42 | 0.96 | 0.93 | 0.97 | 0.97 | **0.99** |
| Human cortex | Adjacent section, random label noise (30%) | 0.28 | **0.85** | 0.84 | **0.85** | **0.85** | 0.84 |
| Human cortex | Donor, random crops | 0.37 | 0.61 | 0.60 | **0.72** | 0.70 | 0.67 |
| Human cortex | Section, random crops | 0.13 | 0.29 | 0.30 | **0.38** | 0.33 | 0.31 |
| Human heart | Patient group | 0.46 | 0.79 | 0.76 | 0.81 | **0.84** | 0.73 |

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


palette = ["#CC79A7", "#D55E00", "#0072B2", "#F0E442", "#009E73"]
colors = dict(zip(sorted(markers), palette))


def plot_slices(samples):
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

<img src="https://raw.githubusercontent.com/edavidk7/friendlyspots/metrics-and-benchmarks/python/figures/README-slices-1.png" alt="" width="75%" style="display: block; margin: auto;" />


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

<img src="https://raw.githubusercontent.com/edavidk7/friendlyspots/metrics-and-benchmarks/python/figures/README-similarity-3.png" alt="" width="50%" style="display: block; margin: auto;" />

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
