from ..reeb.lowerstar import LowerStar 
import matplotlib.pyplot as plt
import numpy as np

class Torus(LowerStar):
    """
    Class to create an example torus using the LowerStar.

    Inherits from:
        LowerStar: A lower star simplicial complex.

    Methods:
        __init__(): Initializes the torus by inserting simplices.
        get_vertex_integer(i, j): Returns the unique integer for the vertex at (i, j).
        get_vertex_coordinates(vertex_int): Returns the (i, j) coordinates for a given vertex integer.
        plot(): Plots a flat representation of the torus's 1-skeleton.

    Example:
        >>> torus = Torus()
        >>> torus.generate_grid(grid_size=5)
        >>> torus.assign_random_values(0, 10)
        >>> torus.plot()

    """
    def __init__(self):
        """Initialize class. Due to SimplexTree C++ weirdness, this does nothing but initialize the instance. Call T.generate_grid(5) to add the content. 
        """
        
        super().__init__()
        
    def generate_grid(self, grid_size = 5):
        """Initializes a simplicial complex representing a torus. The grid_size is used to generate a mesh with (n-1)x(n-1) vertices. Visually, this could be drawn as a grid of nxn vertices, but where the last row and the last column are matched up. 

        Args:
            grid_size (int, optional): _description_. Defaults to 5, Minimum is 4
        """

        if grid_size < 3:
            raise ValueError("grid_size must be at least 4 to form a torus.")
        
        self.grid_size = grid_size
        n = grid_size

        # Add triangles (two per square, with periodic boundary)
        # Note that vertices and edges are added automatically as needed
        
        for i in range(n):
            for j in range(n):
                v0 = self.get_vertex_integer(i, j)
                v1 = self.get_vertex_integer((i+1)%n, j)
                v2 = self.get_vertex_integer(i, (j+1)%n)
                v3 = self.get_vertex_integer((i+1)%n, (j+1)%n)
                # Lower triangle
                self.insert([v0, v1, v2])
                # Upper triangle
                self.insert([v1, v2, v3])
    
    def get_vertex_integer(self, i, j):
        """Returns the unique integer corresponding to the vertex at position (i, j) in the grid.

        Args:
            i (int): Row index.
            j (int): Column index.

        Returns:
            int: Unique integer representing the vertex.
        """
        n = self.grid_size
        return int(i * n + j)
    
    def get_vertex_coordinates(self, vertex_int):
        """Returns the (i, j) coordinates of a vertex given its unique integer.

        Args:
            vertex_int (int): Unique integer representing the vertex.

        Returns:
            tuple: (i, j) coordinates of the vertex.
        """
        n = self.grid_size
        i = vertex_int // n
        j = vertex_int % n
        return (i, j)
    
    def assign_filtration(self, vertex, value):
        """Sets the filtration value for a given vertex. Vertex can be provided either as an integer, as a single integer in a list representing the integer label of the vertex, or as (i, j) coordinates for the grid structure of the torus.

        Args:
            vertex (Union[int, tuple]): Unique integer representing the vertex or (i, j) coordinates.
            value (float): Filtration value to set.
        """
        
        if isinstance(vertex, int):
            vertex = [vertex]
        elif len(vertex) == 1:
            pass
        elif len(vertex) == 2: 
            vertex = [self.get_vertex_integer(*vertex)]
        else:
            raise ValueError("vertex must be an integer, a single-element list, or a tuple of (i, j) coordinates.")
            
        super().assign_filtration(vertex, value)
        
    def draw(self, ax = None, cmap = plt.cm.viridis, **kwargs):
        """Gives a flat plot of the torus's 1-skeleton using matplotlib.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created. Defaults to None.
            cmap (matplotlib.colors.Colormap, optional): Colormap to use for vertex colors based on filtration values. Defaults to plt.cm.viridis.
            **kwargs: Additional keyword arguments passed to plt.scatter for vertex plotting.
            
        """
        
        n = self.grid_size
        if ax is None:
            fig, ax = plt.subplots()

        # Get a grid of vertex positions. Each row is (i,j) in order
        n = self.grid_size  # or set n = 5
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        positions = np.stack([I.ravel(), J.ravel()], axis=1)

        vmin = self.min_filtration()
        vmax = self.max_filtration()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm([self.filtration([v]) for v in range(n*n)]))

        # Plot vertices
        plt.scatter(positions[:,0], positions[:,1], c=colors, s=100, edgecolors='k', zorder = 2, **kwargs)
        # Plot the vertices on the other size, but greyed out to indicate periodicity
        for i in range(n):
            ax.plot(i,n, 'ko', alpha=0.3)  # Top row
            ax.plot(n,i, 'ko', alpha=0.3)  # Right column
        ax.plot(n,n, 'ko', alpha=0.3)  # Top-right corner
        
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Filtration Value')

        # Plot edges
        for simplex in self.get_skeleton(1):
            if len(simplex[0]) == 2:  # Only consider edges
                v0, v1 = simplex[0]
                i0, j0 = self.get_vertex_coordinates(v0)
                i1, j1 = self.get_vertex_coordinates(v1)

                # Handle periodic boundary conditions
                if abs(i1 - i0) > 1:
                    if i0 == 0:
                        i0 += n
                    else:
                        i1 += n
                if abs(j1 - j0) > 1:
                    if j0 == 0:
                        j0 += n
                    else:
                        j1 += n

                ax.plot([i0, i1], [j0, j1], 'k-',zorder = 1)  # 'v-' means violet color, solid line
        
        # Add grey edges on the right and top 
        for i in range(n+1):
            # Top row edges
            ax.plot([i, (i+1)%n], [n, n], 'k--', alpha=0.3, zorder = 1)
            # Right column edges
            ax.plot([n, n], [i, (i+1)%n], 'k--', alpha=0.3, zorder = 1)

        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Flat representation of the Torus 1-skeleton')
        plt.grid(True)
        plt.show()