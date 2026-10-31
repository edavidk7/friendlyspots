# Benchmarks

Scripts comparing the `metric` options of `spatial_similarity()`, plus a
composition-only baseline (label proportions, ignoring where spots are). The
retrieval benchmarks score how well a setting retrieves samples known to belong
together, as nearest-neighbor accuracy (1-NN) and mean average precision (mAP).
The detection benchmark scores how well a setting tells a small biological
alteration apart from label noise, as AUC.

```bash
./download_data.sh     # download and process the public datasets (~2.1 GB)
./run_benchmarks.sh    # run all benchmarks, saving output to results/
```

Both take dataset or benchmark names to run only some, e.g.
`./download_data.sh dlpfc` and `./run_benchmarks.sh synthetic pig`. They use
`../.venv/bin/python` if it exists; set `PYTHON` to use another interpreter.
Processing the heart data requires `h5py`.

| Benchmark | Data | Labels | Retrieval ground truth |
|---|---|---|---|
| `synthetic` | Generated tissue blobs, 6 arrangement classes x 5 samples | 4 labels, composition fixed or varied, optional label noise | Arrangement class |
| `pig` | Durkin et al. (2026), 10 sections (the test data) | Cell types | Time point |
| `dlpfc` | Maynard et al. (2021), 12 sections from [LieberInstitute/HumanPilot](https://github.com/LieberInstitute/HumanPilot) | Manual cortical layers | Donor; adjacent section; also with label noise and random crops |
| `heart` | Kuppe et al. (2022), 15 single-zone sections from [Zenodo 6578047](https://zenodo.org/records/6578047) | Dominant deconvolved cell type | Myogenic (control, remote zone), ischaemic or fibrotic zone |
| `detection` | Synthetic, pig, cortex and heart sections as bases | As above, plus spiked-in alterations of 5-40 spots: a new rare cell type at an interface, or an ectopic domain | Altered vs only noisy (random flips, boundary jitter) |

`python synthetic.py --alternatives` also scores features that were tried and
not added to the package.

Current results are summarised in the
[package README](../README.md#benchmarks).

## Files

- `download_data.sh`: downloads raw files into `data/raw/` (skipping ones
  already there), then runs `prepare_data.py`.
- `prepare_data.py`: processes raw files into `data/<dataset>.csv.gz`, one row
  per spot with `sample`, `array_row`, `array_col`, `label` and the sample's
  ground truth groups. The heart files only have pixel coordinates, so Visium
  array coordinates are recovered by fitting the hexagonal grid.
- `run_benchmarks.sh`: runs benchmarks, saving output to `results/`.
- `common.py`: settings compared, composition baseline, retrieval scoring and
  dataset loading.
- `synthetic.py`, `pig.py`, `dlpfc.py`, `heart.py`, `detection.py`: the benchmarks.

`data/` and `results/` are not tracked by git.
