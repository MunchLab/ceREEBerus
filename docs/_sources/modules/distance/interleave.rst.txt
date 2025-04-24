Interleave class
================

This is the ``Interleave`` class which is used to find a good bound for the interleaving distance between two mapper graphs. 

To get the bound, we can run::

    from cereeberus import Interleave
    myInt = Interleave(M1, M2)
    myInt.fit()

Note that there are two major classes in this module. The ``Interleave`` class is used to compute a bound ``N`` on the interleaving distance between two mapper graphs, while the ``Assignment`` class is used internally to find the best bound ``n+k`` given a fixed value of ``n`` where ``k`` is the computed loss.
A full tutorial can be found in the `Interleaving Basics Jupyter Notebook <../../notebooks/interleaving_basics.ipynb>`_.


.. automodule:: cereeberus.distance.interleave
    :members:
