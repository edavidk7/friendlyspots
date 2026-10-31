# friendlyspots 0.2.0

* New `metric="random_walk"` option for `spatial_similarity()` compares the
  labels each label borders, relative to its own total, discounting how common
  each label is. Its scores don't depend on which other samples are compared.
* New `unit_diagonal` argument scales connection counts to
  C_ij / sqrt(C_ii C_jj) before comparing them, removing how much of each label
  there is.
* Benchmarks comparing the metrics on synthetic and public datasets are in
  `benchmarks/`.
* Now requires `scipy`.

# friendlyspots 0.1.0

* Initial release.
