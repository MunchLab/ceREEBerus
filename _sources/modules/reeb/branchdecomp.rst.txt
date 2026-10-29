BranchDecomp class
==================

The ``BranchDecomp`` class computes and stores a branch decomposition of a Reeb graph.

For a worked example, see the :doc:`branch decomposition tutorial </notebooks/branch_decomp_basics>`.

This implementation and tutorial are based in part on the branch decomposition and
unsmoothing perspective developed in Samira Jabry's thesis: *Unsmoothing of Reeb
Graphs*, Saint Louis University, 2025. Available via ProQuest at
https://www.proquest.com/dissertations-theses/unsmoothing-reeb-graphs/docview/3231769569/se-2.

A branch decomposition breaks a Reeb graph into a collection of non-overlapping upward paths called *branches*. Each branch travels strictly upward in function value, starting at either a local minimum or a saddle point and ending at either a local maximum or another saddle point. The decomposition also records how branch endpoints attach to previously-computed branches, allowing the original Reeb graph to be fully reconstructed.

The decomposition can be computed either directly via this class, or via the convenience method on the Reeb graph::

    import cereeberus
    import cereeberus.data.ex_reebgraphs as ex_rg
    from cereeberus.reeb.branchdecomp import BranchDecomp

    R = ex_rg.dancing_man()

    # Option 1: via the ReebGraph method
    bd = R.branch_decomp()

    # Option 2: directly
    bd = BranchDecomp()
    bd.decompose(R)

    # Visualise the decomposition alongside the original graph
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    R.draw(ax=axes[0])
    bd.draw(ax=axes[1])

    # Reconstruct the original graph
    R2 = bd.reconstruct()

.. automodule:: cereeberus.reeb.branchdecomp
    :members:
