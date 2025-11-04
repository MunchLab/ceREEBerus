# An example lower star filtration built using gudhi's Simplex Tree 

from gudhi import SimplexTree
import numpy as np

class LowerStar(SimplexTree):
    """
    Class to create a simplicial complex that has a lower star filtration, based on Gudhi's SimplexTree.

    Inherits from:
        SimplexTree: A simplex tree structure from the Gudhi library.

    Methods:
        __init__(): Initializes the simplex tree.


    """
    def __init__(self):
        """Initialize class. Due to SimplexTree C++ weirdness, this does nothing but initialize the instance. 
        """
        
        super().__init__()
    
    def assign_filtration(self, vertex, value):
        """Sets the filtration value for a given vertex. Vertex can be provided either as an integer, or as a single integer in a list representing the integer label of the vertex.

        Args:
            vertex (Union[int, tuple]): Name of the vertex (an integer or a single-element list). 
            value (float): Filtration value to set.
        """
        
        if isinstance(vertex, int):
            vertex = [vertex]
        elif len(vertex) != 1:
            raise ValueError("Lower star filtration can only by updated by specificying the function value on a vertex. Input -vertex- must be an integer, or a single-element list.")
            
        super().assign_filtration(vertex, value)
        
        # Set all adjacent simplices to have at least this filtration value
        for simplex, filt in self.get_star(vertex):
            simplex_val = max([self.filtration([v]) for v in simplex])
            super().assign_filtration(simplex, simplex_val)
        
    def max_filtration(self):
        """Returns the maximum filtration value among all simplices in the torus.

        Returns:
            float: Maximum filtration value.
        """
        max_filt = max(f for _, f in self.get_filtration())
        return max_filt

    def min_filtration(self):
        """Returns the minimum filtration value among all simplices in the torus.

        Returns:
            float: Minimum filtration value.
        """
        min_filt = min(f for _, f in self.get_filtration())
        return min_filt
    
    def assign_random_values(self, min, max, seed=None):
        """Assign uniform random filtrations sampled in (min, max) for all vertices.

        Args:
            min (float): Minimum value for random function values
            max (float): Maximum value for random function value
            seed (int, optional): Random seed for reproducibility. Default is None.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
            filtvals = rng.uniform(min, max, self.num_vertices())
        else:
            filtvals = np.random.uniform(min, max, self.num_vertices())
        for v, val in zip(range(self.num_vertices()), filtvals):
            self.assign_filtration(v, val)

    def num_vertices(self):
        """Return the number of vertices (0-simplices) in the simplex tree."""
        return sum(1 for simplex in self.get_simplices() if len(simplex[0]) == 1)
    
    def iter_vertices(self):
        """Yield the names (tuples) of all vertices (0-simplices) in the simplex tree."""
        for simplex in self.get_simplices():
            if len(simplex[0]) == 1:
                yield simplex[0][0]
    
    
if __name__ == "__main__":
    LS = LowerStar()
    LS.insert([0, 1, 2])
    LS.insert([1, 3])
    LS.insert([2,3])
    
    LS.assign_random_values(0.0, 1.0)
    
    for simplex, filt in LS.get_filtration():
        print(f"Simplex: {simplex}, Filtration: {filt}")
