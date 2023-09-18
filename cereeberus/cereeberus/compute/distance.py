def edit(R1, R2):
    """Function to return the edit distance between two Reeb graphs.  Uses the graph_edit_distance function from https://networkx.org/documentation/stable/reference/algorithms/similarity.html.

    Args:
        R1 (reeb graph): Reeb graph or Merge tree
        R2 (reeb graph): Reeb graph or Merge tree

    Returns:
        edit_distance (int): graph edit distance
    """
    import networkx as nx
    edit_distance = nx.graph_edit_distance(R1.G, R2.G)
    return edit_distance