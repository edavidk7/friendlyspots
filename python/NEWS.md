# friendlyspots 0.2.0

* New `metric` options for `spatial_similarity()`, which give scores that don't
  depend on which other samples are compared:
  * `"random_walk"` compares the labels each label borders, relative to its
    own total, discounting how common each label is.
  * `"npmi_heat"` compares how much more or less often labels are near each
    other than their abundance predicts, over a neighborhood set by `t`.
  * `"spectral"` adds to `"random_walk"` a comparison of the labels' graph
    Fourier spectra, capturing structure beyond direct neighbors. It did best
    in the benchmarks.
* New `denoise_labels()` relabels isolated, likely mislabelled spots.
  `spatial_similarity()` applies it by default with `smoothing=0.2`, which
  slightly changes results; use `smoothing=0` to reproduce earlier results.
* Benchmarks comparing the metrics on synthetic and public datasets are in
  `benchmarks/`.
* Now requires `scipy`.

# friendlyspots 0.1.0

* Initial release.
