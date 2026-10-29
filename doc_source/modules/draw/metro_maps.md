# Metro view of interleaving maps

`draw_interleaving_metro_maps` plots the two directions of a solved ceREEBerus
interleaving: `phi: F -> G^n` and `psi: G -> F^n`. The background is the
smoothed destination graph. Each colored strand is one source edge, placed on
the destination edge selected by the ILP's edge matrix. White stations are
destination vertices; outlined stations receive source vertices selected by
the ILP's vertex matrix. Several strands on one edge show shared use, and
divergent strands show different mapped routes.

![Solved torus-to-line ILP maps in both directions](metro_example.png)

```python
from cereeberus import Interleave
from cereeberus.draw.metro_maps import draw_interleaving_metro_maps

result = Interleave(F, G)
result.fit()
fig, axes = draw_interleaving_metro_maps(result, label_stations=False)
fig.savefig("interleaving_metro.png", dpi=200)
```

For maps from `F^n -> G^2n` and `G^n -> F^2n`, pass `key="n"`. A single
direction can be drawn with `draw_metro_map(result, direction="phi")` or
`direction="psi"`. `Assignment.optimize()` and `Assignment.dist_optimize()`
outputs can also be passed directly. The plotting functions do not rerun the
solver or change the graphs.

For `key="n"`, auxiliary map columns at the newly added boundary levels may
be zero in an otherwise feasible ILP solution; these do not create strands.

This is a deliberately direct view of the ILP matrices, rather than a full
MetroSets layout optimizer. A source edge is one strand; no path decomposition
or uncertainty measure is inferred. Parallel edges with identical endpoint
labels inherit the current matrix labeling and may share a drawn centerline.

Design references: [MetroSets](https://arxiv.org/abs/2008.09367),
[Topological Characterization and Uncertainty Visualization of Atmospheric Rivers](https://www.sci.utah.edu/~beiwang/publications/AR_BeiWang_2024.pdf), and
[Uncertainty Visualization for Approximated Barycenter of Contour Trees](https://www.sci.utah.edu/~beiwang/publications/Uncertainty_CT_BeiWang_2026.pdf).

For a worked, executed example, open the [metro interleaving notebook](../../notebooks/metro_interleaving_maps.ipynb).
