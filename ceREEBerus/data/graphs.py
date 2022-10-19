def reeb_torus():
    """ Reeb graph of a torus
        Args:

        Returns:
            reeb_graph (networkx graph): reeb graph of a torus

    """
    import networkx as nx
    G = nx.Graph()
    G.add_node(1,pos=(1,1))
    G.add_node(2,pos=(1,2))
    G.add_edge(1,2)
    G.add_node(3,pos=(0,3))
    G.add_node(4,pos=(2,3))
    G.add_edge(2,3)
    G.add_edge(2,4)
    G.add_node(5,pos=(1,4))
    G.add_edge(5,3)
    G.add_edge(5,4)
    G.add_node(6,pos=(1,5))
    G.add_edge(5,6)

    return G