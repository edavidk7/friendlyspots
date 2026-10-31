# friendlyspots 0.2.0

* New `metric` options for `spatial_similarity()`, which give scores that don't
  depend on which other samples are compared:
  * `"random_walk"` compares the labels each label borders, relative to its
    own total, discounting how common each label is.
  * `"moran"` compares how far each pair of labels' joins are from random
    placement of the sample's own labels, as Moran's join-count z-scores, by
    the angle between samples' z-score vectors. `exact_tail=True` uses exact
    tail probabilities for label pairs with few expected joins.
* New `unit_diagonal` argument scales connection counts to
  C_ij / sqrt(C_ii C_jj) before comparing them, removing how much of each label
  there is.
* Benchmarks comparing the metrics on synthetic and public datasets are in
  `benchmarks/`.
* Now requires `scipy`.

# friendlyspots 0.1.0

* Initial release.
