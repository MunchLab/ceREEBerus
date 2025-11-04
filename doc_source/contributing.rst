Info on contributing to ceREEBerus 
-----------------------------------

We love having new contributors! Please read the following guidelines to help you get started.

How to Contribute: Forking and Pull Requests
=============================================

1. Visit the GitHub repository: https://github.com/MunchLab/ceREEBerus
2. Click "Fork" in the top right to create your own copy of the repository.
3. Clone your fork to your local machine:

    .. code-block:: bash

        git clone https://github.com/<your-username>/ceREEBerus.git

4. Create a new branch for your changes:

    .. code-block:: bash

        git checkout -b my-feature-branch

5. Make your changes and commit them with clear messages.
6. Push your branch to your fork:

    .. code-block:: bash

        git push origin my-feature-branch

7. Go to https://github.com/MunchLab/ceREEBerus and click "Compare & pull request" to open a pull request. 
8. Fill out the PR template and submit your pull request for review.

For more details, see GitHub's guide: https://docs.github.com/en/get-started/quickstart/contributing-to-projects

PR Review Checklist
~~~~~~~~~~~~~~~~~~~~~~~

Before we approve any pull request, we'll ask you to do the following things: 

- Ensure your code follows the code style of this project. You can run:

    .. code-block:: bash

        make format

to autoformat your code using `black <https://black.readthedocs.io/en/stable/>`_.

If your change requires a change to the documentation, please update the documentation as necessary. You can compile the docs locally by running:

    .. code-block:: bash

        make docs

to ensure everything looks good. You will be able to view your changes in the built documentation by opening the `docs/_build/html/index.html` file in your web browser. Do not add the Docs folder to your git repository. The final documentation will be built on github when your PR is merged.
 ``make docs``: Build the documentation in the docs/ folder
- Add tests to cover your changes. You can run:

    .. code-block:: bash

        make tests

to run the unit tests and ensure everything is passing. If the tests fail, we won't be able to merge your PR.

- Please make sure to increment the version number in the ``pyproject.toml`` file if your changes require a new release to be pushed to PyPI. If the version number is not incremented, the package will not be pushed to PyPI, which is useful if your PR is only for updating documentation.

Useful Makefile Commands
========================

The Makefile in this project provides several helpful commands for development and documentation:

- ``make install-requirements``: Install all dependencies from requirements.txt, including those needed for development and documentation.
- ``make install-editable``: Install the package in editable mode for development
- ``make lint``: Run `flake8 <https://flake8.pycqa.org/en/latest/>`_ linter on the code
- ``make format``: Format code using `black <https://black.readthedocs.io/en/stable/>`_
- ``make tests``: Run the unit tests
- ``make html``: Build the documentation in the docs/ folder
- ``make release``: Build the package for release
- ``make clean``: Autoformat the code using autopep8
- ``make all``: Install requirements, install ceREEBerus in editable mode, then run format, tests, and docs
- ``make help``: See a summary of all available commands.

Info on documentation
=====================

Markdown vs. reStructuredText
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project uses sphinx with reStructuredText for documentation. Here is a `cheat sheet <https://sphinx-tutorial.readthedocs.io/cheatsheet/>`_ for reStructuredText. While markdown also works, it tends to be less powerful for certain things like equations.

Basic Examples
~~~~~~~~~~~~~~~~

Here are some minimal working examples. 
See `sphinx-rtd-theme documentation <https://sphinx-rtd-theme.readthedocs.io/en/stable/demo/api.html>`_ for full information. 
This is a docstring example for a method::
    

    """
    Returns the subgraph of the Reeb graph with image in the open interval :math:`(a, b)`.
    This will convert any edges that cross the slice into vertices.

    Parameters:
        a (float): The lower bound of the slice :math:`a`.
        b (float): The upper bound of the slice :math:`b`.

    Returns:
        ReebGraph: The subgraph of the Reeb graph with image in :math:`(a, b)`.
    """

Style choices we've made 
~~~~~~~~~~~~~~~~~~~~~~~~

There is lots of vocabulary around TDA constructions, and lots of opinions about coding styles. Here are some style choices we've made for this project. Are we consistent with them? Not always, but we try!

- Everything is a "vertex" or an "edge" (not a "node" or a "link")
- File names are all lowercase with underscores (e.g., reeb_graph.py)
- Class names are in CamelCase (e.g., ReebGraph)
- Method and function names are in ``snake_case``  or ``camelCase`` starting lower case (e.g., ``compute_reeb_graph`` or ``computeReeb``)
