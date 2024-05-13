Info on contributing to the project 
-----------------------------------

Info on Markdown vs. reStructuredText
=====================================

This project uses reStructuredText for documentation. Here is a `cheat sheet <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ for reStructuredText. However, markdown also works. It's not the greatest though so eventually I need to (TODO) write up how it works here. 

Info on documentation
=====================

This stuff is copied from the `sphinx-rtd-theme documentation <https://sphinx-rtd-theme.readthedocs.io/en/stable/demo/api.html>`_. Here are some minimal working examples. 
This is a docstring example for a method::
    
    """
    Returns the subgraph of the Reeb graph with image in the open interval (a,b).
    This will convert any edges that cross the slice into vertices.

    Parameters:
        a (float): The lower bound of the slice.
        b (float): The upper bound of the slice.
    
    Returns:
        ReebGraph: The subgraph of the Reeb graph with image in (a,b).
    """

Things I always forget
=======================

- TODO: How to write equations 
- TODO: How to write links 
- TODO: How to include images 
- TODO: How to include code snippets

