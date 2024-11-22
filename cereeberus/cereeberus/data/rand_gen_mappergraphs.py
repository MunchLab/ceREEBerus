
from cereeberus import MapperGraph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def set_global_seed(seed):
    """Set the global random seed.
    
    Parameters:
    seed (int): The seed to set.
    """

    global GLOBAL_SEED
    GLOBAL_SEED = seed
    random.seed(GLOBAL_SEED)

def generate_heights(length):
    """Generate a range of heights.
    
    Parameters:
    length (int): The number of heights to generate.
    
    Returns:
    list: A list of heights.
    """
    return range(1, length + 1)

def create_node_dict(width, heights):
    """
    Create a dictionary of nodes with their heights. Ensures at least one node exists at each height.
    
    Parameters:
    width (int): The maximum number of nodes that can exist at a height.
    heights (list): A list of heights.

    Returns:
    dict: A dictionary of nodes with their heights.
   
    """
    node_dict = {}
    for h in heights:
        num_nodes = random.randint(1, width)
        for i in range(num_nodes):
            node_dict[f"{h}_{i}"] = h
    return node_dict

def create_edge_list(node_dict, cutoff):
    """
    Create edges between nodes from adjacent heights. Edges are added based on the cutoff probability.
    
    Parameters:
    node_dict (dict): A dictionary of nodes with their heights.
    cutoff (float): The probability of adding an edge between nodes from adjacent heights.

    Returns:
    list: A list of edges.
    """
    height_dict = {}
    for node, height in node_dict.items():
        height_dict.setdefault(height, []).append(node)

    edge_list = []
    for height, nodes in height_dict.items():
        for adj_height in (height - 1, height + 1):
            if adj_height in height_dict:
                adj_nodes = height_dict[adj_height]
                for node1 in nodes:
                    edge_list.extend(
                        (node1, node2) for node2 in adj_nodes if random.random() <= cutoff
                    )
    return edge_list

def create_graph(width, length, cutoff, verbose=False, seed=9):
    """
    Create a connected NetworkX graph with nodes and edges based on parameters.

    Parameters:
    width (int): The maximum number of nodes that can exist at a height.
    length (int): The number of heights to generate.
    cutoff (float): The probability of adding an edge between nodes from adjacent heights.
    verbose (bool): Whether to print the nodes and edges of the graph.
    seed (int): The seed to set.

    Returns:
    nx.Graph: A connected NetworkX graph.
    """
    set_global_seed(seed)

    while True:
        heights = generate_heights(length)
        node_dict = create_node_dict(width, heights)
        edge_list = create_edge_list(node_dict, cutoff)

        G = nx.Graph()
        G.add_nodes_from((node, {'func_val': value}) for node, value in node_dict.items())
        G.add_edges_from(edge_list)

        # Exit the loop if the graph is connected
        if nx.is_connected(G):
            break

    if verbose:
        pos = {node: (int(node.split('_')[1]), node_dict[node]) for node in G.nodes()}
        print(f"Nodes: {G.nodes()}")
        print(f"Edges: {G.edges()}")
        nx.draw(G, pos, with_labels=True)

    return G, node_dict


def create_mapper_graph(width, length, cutoff, verbose=False, seed=9):
    """
    Create a MapperGraph object with nodes and edges based on parameters.

    Parameters:
    width (int): The maximum number of nodes that can exist at a height.
    length (int): The number of heights to generate.
    cutoff (float): The probability of adding an edge between nodes from adjacent heights.
    verbose (bool): Whether to print the nodes and edges of the graph.
    seed (int): The seed to set.

    Returns:
    MapperGraph: A MapperGraph object.
    """
    G, node_dict = create_graph(width, length, cutoff, verbose, seed)

    mapper_graph = MapperGraph(G,node_dict)
    return mapper_graph