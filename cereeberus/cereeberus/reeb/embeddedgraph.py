import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cereeberus import ReebGraph


class EmbeddedGraph(nx.Graph):
    """
    A class to represent a graph with 2D embedded coordinates for each vertex.

    Attributes
        graph : nx.Graph
            a NetworkX graph object
        coordinates : dict
            a dictionary mapping vertices to their (x, y) coordinates

    """

    def __init__(self):
        """
        Initializes an empty EmbeddedGraph object.

        """
        super().__init__()
        self.coordinates = {}

    def add_node(self, vertex, x, y):
        """
        Adds a vertex to the graph and assigns it the given coordinates.

        Parameters:
            vertex (str):
                The vertex to be added.
            x (float):
                The x-coordinate of the vertex.
            y (float):
                The y-coordinate of the vertex.

        """
        super().add_node(vertex)
        self.coordinates[vertex] = (x, y)

    def add_nodes_from(self, nodes, coordinates):
        """
        Adds multiple vertices to the graph and assigns them the given coordinates.

        Parameters:
            nodes (list):
                A list of vertices to be added.
            coordinates (dict):
                A dictionary mapping vertices to their coordinates.

        """
        super().add_nodes_from(nodes)
        self.coordinates.update(coordinates)

    def add_edge(self, u, v):
        """
        Adds an edge between the vertices u and v if they exist.

        Parameters:
            u (str):
                The first vertex of the edge.
            v (str):
                The second vertex of the edge.

        """
        if not self.has_node(u) or not self.has_node(v):
            raise ValueError("One or both vertices do not exist in the graph.")
        else:
            super().add_edge(u, v)

    def get_coordinates(self, vertex):
        """
        Returns the coordinates of the given vertex.

        Parameters:
            vertex (str):
                The vertex whose coordinates are to be returned.

        Returns:
            tuple: The coordinates of the vertex.

        """
        return self.coordinates.get(vertex)

    def set_coordinates(self, vertex, x, y):
        """
        Sets the coordinates of the given vertex.

        Parameters:
            vertex (str):
                The vertex whose coordinates are to be set.
            x (float):
                The new x-coordinate of the vertex.
            y (float): 
                The new y-coordinate of the vertex.

        Raises:
            ValueError: If the vertex does not exist in the graph.

        """
        if vertex in self.coordinates:
            self.coordinates[vertex] = (x, y)
        else:
            raise ValueError("Vertex does not exist in the graph.")

    def get_bounding_box(self):
        """
        Method to find a bounding box of the vertex coordinates in the graph.

        Returns:
            list: A list of tuples representing the minimum and maximum x and y coordinates.

        """
        if not self.coordinates:
            return None

        x_coords, y_coords = zip(*self.coordinates.values())
        return [(min(x_coords), max(x_coords)), (min(y_coords), max(y_coords))]

    def get_bounding_radius(self):
        """
        Method to find the radius of the bounding circle of the vertex coordinates in the graph.

        Returns:
            float: The radius of the bounding circle.

        """
        if not self.coordinates:
            return 0

        x_coords, y_coords = zip(*self.coordinates.values())
        norms = [np.linalg.norm(point) for point in zip(x_coords, y_coords)]

        return max(norms)

    def get_mean_centered_coordinates(self):
        """
        Method to find the mean-centered coordinates of the vertices in the graph.

        Returns:
            dict: A dictionary mapping vertices to their mean-centered coordinates.

        """
        if not self.coordinates:
            return None

        x_coords, y_coords = zip(*self.coordinates.values())
        mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)

        return {v: (x - mean_x, y - mean_y) for v, (x, y) in self.coordinates.items()}

    def set_mean_centered_coordinates(self):
        """
        Method to set the mean-centered coordinates of the vertices in the graph. Warning: This overwrites the original coordinates

        """

        self.coordinates = self.get_mean_centered_coordinates()

    def g_omega(self, theta):
        """
        Function to compute the function :math:`g_\omega(v)` for all vertices :math:`v` in the graph in the direction of :math:`\\theta \in [0,2\pi]` . This function is defined by :math:`g_\omega(v) = \langle \\texttt{pos}(v), \omega \\rangle` .

        Parameters:

            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to compute the :math:`g(v)` values.

        Returns:

            dict: A dictionary mapping vertices to their :math:`g(v)` values.

        """

        omega = (np.cos(theta), np.sin(theta))

        g = {}
        for v in self.nodes:
            g[v] = np.dot(self.coordinates[v], omega)
        return g

    def g_omega_edges(self, theta):
        """
        Calculates the function value of the edges of the graph by making the value equal to the max vertex value 

        Parameters:

            theta (float): 
                The direction of the function to be calculated.

        Returns:
            dict
                A dictionary of the function values of the edges.
        """
        g = self.g_omega(theta)

        g_edges = {}
        for e in self.edges:
            g_edges[e] = max(g[e[0]], g[e[1]])

        return g_edges

    def sort_vertices(self, theta, return_g=False):
        """
        Function to sort the vertices of the graph according to the function g_omega(v) in the direction of theta \in [0,2*np.pi].

        Parameters:
            theta (float):
                The angle in [0,2*np.pi] for the direction to sort the vertices.
            return_g (bool):
                Whether to return the g(v) values along with the sorted vertices.

        Returns:
            list
                A list of vertices sorted in increasing order of the :math:`g(v)` values. 
                If ``return_g`` is True, also returns the ``g`` dictionary with the function values ``g[vertex_name]=func_value``. 

        """
        g = self.g_omega(theta)

        v_list = sorted(self.nodes, key=lambda v: g[v])

        if return_g:
            # g_sorted = [g[v] for v in v_list]
            return v_list, g
        else:
            return v_list

    def sort_edges(self, theta, return_g=False):
        """
        Function to sort the edges of the graph according to the function

        .. math ::

            g_\omega(e) = \max \{ g_\omega(v) \mid  v \in e \}

        in the direction of :math:`\\theta \in [0,2\pi]` .

        Parameters:
            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to sort the edges.
            return_g (bool):
                Whether to return the :math:`g(v)` values along with the sorted edges.

        Returns:
            A list of edges sorted in increasing order of the :math:`g(v)` values. 
            If ``return_g`` is True, also returns the ``g`` dictionary with the function values ``g[vertex_name]=func_value``. 

        """
        g_e = self.g_omega_edges(theta)

        e_list = sorted(self.edges, key=lambda e: g_e[e])

        if return_g:
            # g_sorted = [g[v] for v in v_list]
            return e_list, g_e
        else:
            return e_list

    def lower_edges(self, v, omega):
        """
        Function to compute the number of lower edges of a vertex v for a specific direction (included by the use of sorted v_list).

        Parameters:
            v (str):
                The vertex to compute the number of lower edges for.
            omega (tuple): 
                The direction vector to consider given as an angle in [0, 2pi].

        Returns:
            int: The number of lower edges of the vertex v.

        """
        L = [n for n in self.neighbors(v)]
        gv = np.dot(self.coordinates[v], omega)
        Lg = [np.dot(self.coordinates[v], omega) for v in L]
        return sum(n >= gv for n in Lg)  # includes possible duplicate counts

    def plot(self, bounding_circle=False, color_nodes_theta=None, ax=None, **kwargs):
        """
        Function to plot the graph with the embedded coordinates.

        If ``bounding_circle`` is True, a bounding circle is drawn around the graph.

        If ``color_nodes_theta`` is not None, it should be given as a theta in :math:`[0,2\pi]`. Then the nodes are colored according to the :math:`g(v)` values in the direction of theta.

        """
        if ax is None:
            fig, ax = plt.subplots()
            # print("making new figure")
        else:
            fig = ax.get_figure()

        pos = self.coordinates
        if color_nodes_theta == None:
            nx.draw(self, pos, with_labels=True, ax=ax, **kwargs)
        else:
            g = self.g_omega(color_nodes_theta)
            color_map = [g[v] for v in self.nodes]
            # Some weird plotting to make the colorbar work.
            pathcollection = nx.draw_networkx_nodes(
                self, pos, node_color=color_map, ax=ax)
            nx.draw_networkx_labels(self, pos=pos, font_color='black', ax=ax)
            nx.draw_networkx_edges(self, pos, ax=ax, width=1, **kwargs)
            fig.colorbar(pathcollection, ax=ax, **kwargs)

        plt.axis('on')
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)

        if bounding_circle:
            r = self.get_bounding_radius()
            ax = plt.gca()
            circle1 = plt.Circle((0, 0), r, fill=False,
                                 linestyle='--', color='r')
            ax.add_patch(circle1)
            plt.axis('square')

        return ax



    def reeb_graph_from_direction(self, theta):
        """
        Function to create a ReebGraph from a given direction theta.

        Parameters:
            theta (float):
                The direction in [0, 2pi] to calculate the ReebGraph.

        Returns:
            ReebGraph: The ReebGraph object created from the EmbeddedGraph.

        """

        sorted_verts, g_verts= self.sort_vertices(theta, return_g = True)
        g_vert_list = np.array([g_verts[v] for v in sorted_verts])

        # Get the locations of a new function value since those will collapse down to a single vertex in the Reeb graph
        new_val_locs = np.where(1-np.isclose(g_vert_list[:-1] , g_vert_list[1:]))[0] +1
        new_val_locs = np.concatenate([[0,], new_val_locs])

        # This dictionary will be find_comp[vertex_name] = new_vertex_name
        find_comp = {}
        vert_list = []
        f_dict_new = {}

        for val_i,i in enumerate(new_val_locs):

            

            if val_i == len(new_val_locs)-1:
                verts = sorted_verts[i:]
            else:
                verts = sorted_verts[i:new_val_locs[val_i+1]]

            L = list(nx.connected_components(nx.induced_subgraph(self,verts)))
            for cc_list in L:
                new_vert = '-'.join(cc_list)
                vert_list.append(new_vert)
                for v in cc_list:
                    find_comp[v] = new_vert
                f_dict_new[new_vert] = g_vert_list[i]

        R = ReebGraph()
        R.add_nodes_from(vert_list,f_dict_new)

        for e in self.edges:
            if find_comp[e[0]] != find_comp[e[1]]:
                R.add_edge(find_comp[e[0]],find_comp[e[1]])
        
        return R




if __name__ == "__main__":
    # Example usage of the EmbeddedGraph class

    # Create an instance of the EmbeddedGraph class
    graph = EmbeddedGraph()

    # Add vertices with their coordinates
    graph.add_node('A', 1, 2)
    graph.add_node('B', 3, 4)
    graph.add_node('C', 5, 6)

    # Add edges between vertices
    graph.add_edge('A', 'B')
    graph.add_edge('B', 'C')

    # Get coordinates of a vertex
    coords = graph.get_coordinates('A')
    print(f'Coordinates of A: {coords}')

    # Set new coordinates for a vertex
    graph.set_coordinates('A', 7, 8)
    coords = graph.get_coordinates('A')
    print(f'New coordinates of A: {coords}')

    # Get the bounding box of the vertex coordinates
    bbox = graph.get_bounding_box()
    print(f'Bounding box: {bbox}')
