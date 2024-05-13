
#=================================
# Construct an example Reeb
# graph of a simple upright torus
#=================================

import networkx as nx

def torus_graph():
    """ nx graph input to construct the example Reeb graph of a torus

        Returns:
            graph (networkx graph): reeb graph of a torus

    """

    G = nx.MultiGraph()
    G.add_node(0,pos=(1, 1))
    G.add_node(1,pos=(1, 2))
    G.add_edge(0,1)

    G.add_node(2,pos=(.5,3))
    G.add_node(3,pos=(1.5,3))
    G.add_edge(1,2)
    G.add_edge(1,3)

    G.add_node(4,pos=(1,4))
    G.add_edge(4,2)
    G.add_edge(4,3)

    G.add_node(5,pos=(1,5))
    G.add_edge(4,5)

    fx = {0: 1, 1: 2, 2: 3, 3: 3, 4: 4, 5: 5}
    nx.set_node_attributes(G, fx, 'fx')

    return G

def reeb_torus_no_fx():
    """ 
    Reeb graph of a torus with no function values

        Returns:
            graph (networkx graph): reeb graph of a torus

    """

    G = nx.MultiGraph()
    G.add_node(0,pos=(1,1))
    G.add_node(1,pos=(1,2))
    G.add_edge(0,1)

    G.add_node(2,pos=(.5,3))
    G.add_node(3,pos=(1.5,3))
    G.add_edge(1,2)
    G.add_edge(1,3)

    G.add_node(4,pos=(1,4))
    G.add_edge(4,2)
    G.add_edge(4,3)

    G.add_node(5,pos=(1,5))
    G.add_edge(4,5)

    return G

#=================================
# Construct some of Liz's other
# favorite example Reeb graphs
#=================================

def dancing_man():
    """ nx graph input to construct the example dancing man

        Returns:
            graph (networkx graph): reeb graph of dancing man

    """

    G = nx.MultiGraph()
    G.add_node(0,pos=(3,7))
    G.add_node(1,pos=(3,6))
    G.add_edge(0,1) 

    G.add_node(2,pos=(1,5))
    G.add_edge(1,2)

    G.add_node(3, pos=(5,5))
    G.add_edge(1,3)

    G.add_node(4, pos=(7,6))
    G.add_edge(3,4)

    G.add_node(5, pos=(0,4))
    G.add_edge(5,2)

    G.add_node(6, pos=(3,4))
    G.add_edge(6,2)
    G.add_edge(6,3)

    G.add_node(7, pos=(3,1))
    G.add_edge(7,6)

    dd = {0: 1, 1: 2, 2: 2, 3: 1, 4:1, 5:0, 6:1, 7:0}
    nx.set_node_attributes(G, dd, 'down_deg')

    ud = {0: 0, 1: 1, 2: 1, 3: 2, 4: 0, 5:1, 6:2, 7:1}
    nx.set_node_attributes(G, ud, 'up_deg')

    fx = {0: 7, 1: 6, 2: 5, 3: 5, 4: 6, 5: 4, 6: 4, 7:1}
    nx.set_node_attributes(G, fx, 'fx')

    return G

def juggling_man():
    """ Dancing Man Graph with isolates

        Returns:
            graph (networkx graph): reeb graph of juggling man, which is dancing man with isolates
    """

    G = nx.MultiGraph()
    G.add_node(0,pos=(3,7))
    G.add_node(1,pos=(3,6))
    G.add_edge(0,1) 

    G.add_node(2,pos=(1,5))
    G.add_edge(1,2)

    G.add_node(3, pos=(5,5))
    G.add_edge(1,3)

    G.add_node(4, pos=(7,6))
    G.add_edge(3,4)

    G.add_node(5, pos=(0,4))
    G.add_edge(5,2)

    G.add_node(6, pos=(3,4))
    G.add_edge(6,2)
    G.add_edge(6,3)

    G.add_node(7, pos=(3,1))
    G.add_edge(7,6)

    G.add_node(8, pos=(6.5,6.5))
    G.add_node(9, pos=(2,6.5))
    G.add_node(10, pos=(.75,5.5))

    dd = {0: 1, 1: 2, 2: 2, 3: 1, 4:1, 5:0, 6:1, 7:0, 8:0, 9:0, 10:0}
    nx.set_node_attributes(G, dd, 'down_deg')

    ud = {0: 0, 1: 1, 2: 1, 3: 2, 4: 0, 5:1, 6:2, 7:1, 8:0, 9:0, 10:0}
    nx.set_node_attributes(G, ud, 'up_deg')

    fx = {0: 7, 1: 6, 2: 5, 3: 5, 4: 6, 5: 4, 6: 4, 7:1, 8:6.5, 9:6.5, 10:5.5}
    nx.set_node_attributes(G, fx, 'fx')

    return G

def simple_loops():
    """ Simple loops example for plotting loops

        Returns:
            graph (networkx graph): reeb graph with simple loops
    """

    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0,1) 
    G.add_edge(0,1)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(2,3)

    dd = {0: 0, 1: 2, 2: 1, 3: 2}
    nx.set_node_attributes(G, dd, 'down_deg')

    ud = {0: 2, 1: 1, 2: 2, 3: 0}
    nx.set_node_attributes(G, ud, 'up_deg')

    fx = {0: 0, 1: 1, 2: 2, 3: 3}
    nx.set_node_attributes(G, fx, 'fx')

    return G

def simple_loops_unordered():
    """ Simple loops example for plotting loops and testing with unordered edges

        Returns:
            graph (networkx graph): reeb graph with unordered edges
    """

    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0,1) 
    G.add_edge(1,0)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(3,2)

    dd = {0: 0, 1: 2, 2: 1, 3: 2}
    nx.set_node_attributes(G, dd, 'down_deg')

    ud = {0: 2, 1: 1, 2: 2, 3: 0}
    nx.set_node_attributes(G, ud, 'up_deg')

    fx = {0: 0, 1: 1, 2: 2, 3: 3}
    nx.set_node_attributes(G, fx, 'fx')

    return G



