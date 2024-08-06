
# %--- Example usage of the EmbeddedGraph class ---%

from cereeberus import EmbeddedGraph

def exampleEmbeddedGraph(mean_centered=True):
    """
    Function to create an example ``EmbeddedGraph`` object. Helpful for testing.

    Returns:
        EmbeddedGraph: An example ``EmbeddedGraph`` object.

    """
    graph = EmbeddedGraph()

    graph.add_node('A', 1, 2)
    graph.add_node('B', 3, 4)
    graph.add_node('C', 5, 7)
    graph.add_node('D', 3, 6)
    graph.add_node('E', 4, 3)
    graph.add_node('F', 4, 5)

    graph.add_edge('A', 'B')
    graph.add_edge('B', 'C')
    graph.add_edge('B', 'D')
    graph.add_edge('B', 'E')
    graph.add_edge('C', 'D')
    graph.add_edge('E', 'F')

    if mean_centered:
        graph.set_mean_centered_coordinates()

    return graph