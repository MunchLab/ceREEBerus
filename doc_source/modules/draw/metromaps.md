# Metro view of interleaving maps

`draw_metromaps` takes a fitted `Interleave` object and returns a Matplotlib
figure. It reads the Mapper graphs, their smoothings, and the vertex and edge
maps selected by the ILP. It does not run the optimization or alter the maps.

The default figure has two rows. The top shows the `phi` map, with $F$ beside
$G^n$; the bottom shows the `psi` map, with $G$ beside $F^n$. A source edge and
its mapped target strand have the same color. By default, successive edges
between an endpoint or junction share one branch color. Set `color_by="edge"`
to distinguish every edge. Target station labels show the mapped source
vertices. Node labels follow `ReebGraph.draw()` (`s0`, `r0`, and so on).

![Paired maps from a solved loop-to-line interleaving](metro_example.png)

```python
from cereeberus import Interleave
from cereeberus.data.ex_mappergraphs import line, torus
from cereeberus.draw.metromaps import draw_metromaps

F = torus(0, 2, 10, 12, delta=1, seed=17)
G = line(0, 12, seed=5)
result = Interleave(F, G)
result.fit()

figure = draw_metromaps(result)                     # both map directions
phi_figure = draw_metromaps(result, direction="phi") # F beside G^n
edge_figure = draw_metromaps(result, color_by="edge")
```

Use `direction="psi"` for just $G$ beside $F^n$. Set `key="n"` to view the
auxiliary maps $F^n\to G^{2n}$ and $G^n\to F^{2n}$. The function returns the
figure; its panels are available through `figure.axes`.

Each source edge remains a separate ILP assignment. Branch coloring only groups
adjacent edges visually; it does not infer a branch-level interleaving or an
uncertainty value. Parallel edges with identical endpoint labels may share a
drawn centerline. This is a direct plot of ceREEBerus outputs, inspired by the
metro-map visual language of [MetroSets](https://arxiv.org/abs/2008.09367),
not an implementation of its layout algorithms.

For a worked example, open the [metro interleaving notebook](../../notebooks/metro_interleaving_maps.ipynb).

Other visual references: [Topological Characterization and Uncertainty Visualization of Atmospheric Rivers](https://www.sci.utah.edu/~beiwang/publications/AR_BeiWang_2024.pdf) and [Uncertainty Visualization for Approximated Barycenter of Contour Trees](https://www.sci.utah.edu/~beiwang/publications/Uncertainty_CT_BeiWang_2026.pdf).
