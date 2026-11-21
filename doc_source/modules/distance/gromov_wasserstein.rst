Gromov-Wasserstein distance
===========================

This module compares two Reeb graphs with the Gromov-Wasserstein distance, following Chambers and Meng, *Reeb Graph Comparison via the Gromov-Wasserstein Distance*. Each Reeb graph is turned into a metric-measure space:

- the metric on the vertices is the **Symmetric Reeb Radius**, the average of the two directed Reeb radii of Curry et al.;
- the probability measure on the vertices comes from the **Persistence Image** of the extended persistence diagram of the Reeb graph.

The distance is then computed with the `POT <https://pythonot.github.io/>`_ library. To compare two Reeb graphs (or Mapper graphs), run::

    from cereeberus import GromovWasserstein
    myGW = GromovWasserstein(R1, R2, sigma=0.5)
    myGW.fit()

After ``fit``, the optimal transport plan between the vertices is stored in ``myGW.plan``. The metric and the measure can also be computed on their own with :func:`~cereeberus.distance.gromov_wasserstein.symmetric_reeb_radius` and :func:`~cereeberus.distance.gromov_wasserstein.pi_measure`.

A full tutorial can be found in the `Gromov-Wasserstein Basics Jupyter Notebook <../../notebooks/gromov_wasserstein_basics.ipynb>`_.


.. automodule:: cereeberus.distance.gromov_wasserstein
    :members:
