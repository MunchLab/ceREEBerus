from .unionfind import UnionFind
from ..reeb.lowerstar import LowerStar
import numpy as np



def is_face(sigma, tau):
    '''
    Check if tau is a face of sigma
    
    Args:
        sigma: A simplicial complex (a list of simplices).
        tau: A simplex (a list of vertices).

    Returns:
        bool: True if tau is a face of sigma, False otherwise.
    '''
    return set(tau).issubset(set(sigma))

def get_levelset_components(L):
    '''
    Given a list of simplices L representing a level set, compute the connected components. This function is really only helpful inside of computeReeb.
    
    Args:
        L: A list of simplices (each simplex is a list of vertices).

    Returns:
        dict: A dictionary where keys are representative simplices and values are lists of simplices in the same connected component.
    '''
    
    UF = UnionFind(range(len(L)))
    for i, simplex1 in enumerate(L):
        for j, simplex2 in enumerate(L):
            if i < j:
                # Check if they share a vertex
                if is_face(simplex1, simplex2) or is_face(simplex2, simplex1):
                    UF.union(i, j)
    
    # Replace indices with simplices 
    components_index = UF.components_dict()
    components = {}
    for key in components_index:
        components[tuple(L[key])] = [L[i] for i in components_index[key]]
    
    
    return components


def computeReeb(K: LowerStar, verbose = False):
    """Computes the Reeb graph of a Lower Star Simplicial Complex K.

    Args:
        K (LowerStar): A Lower Star Simplicial Complex with assigned filtration values.
        verbose (boolean): Make it True if you want lots of printouts.

    Returns:
        ReebGraph: The computed Reeb graph.

    Example:
        >>> from cereeberus.reeb.LowerStar import LowerStar
        >>> K = LowerStar()
        >>> K.insert([0, 1, 2])
        >>> K.insert([1, 3])
        >>> K.insert([2,3])
        >>> K.assign_filtration([0], 0.0)
        >>> K.assign_filtration([1], 3.0)
        >>> K.assign_filtration([2], 5.0)
        >>> K.assign_filtration([3], 7)
        >>> R = computeReeb(K)
        >>> R.draw()
    """
    from ..reeb.reebgraph import ReebGraph
    
    funcVals = [(i,K.filtration([i])) for i in K.iter_vertices()]  
    funcVals.sort(key=lambda x: x[1])  # Sort by filtration value    
    
    R = ReebGraph() 

    currentLevelSet = []
    components = {}
    half_edge_index = 0

    # This will keep track of the components represented by every vertex in the graph so far. 
    # It will be vertName: connected_component (given as a list of lists) represented by that vertex 
    vert_to_component = {}

    edges_at_prev_level = []


    for i, (vert, filt) in enumerate(funcVals):
        if verbose:
            print(f"\n---\n Processing {vert} at func val {filt:.2f}")
        now_min = filt
        now_max = funcVals[i+1][1] if i+1 < len(funcVals) else np.inf
        star = K.get_star([vert])
        lower_star = [s[0] for s in star if s[1] <= filt and len(s[0]) > 1]
        upper_star = [s[0] for s in star if s[1] > filt and len(s[0]) > 1]
        
        if verbose:
            print(f"  Lower star simplices: {lower_star}")
            print(f"  Upper star simplices: {upper_star}")
        
        #----
        # Update the levelset list 
        #----
        
        for s in lower_star:
            # Remove from current level set
            if s in currentLevelSet:
                currentLevelSet.remove(s)
            
        currentLevelSet.append([vert])  # Add the vertex itself to the level set
        components_at_vertex = get_levelset_components(currentLevelSet)
        
        if verbose:
            print(f"  Current level set simplices: {currentLevelSet}")
            print(f"  Level set components at vertex {vert} (func val {filt:.2f}):")
            for comp in components_at_vertex.values():
                print(f"    Component: {comp}")
        
        verts_at_level = []
        for rep, comp in components_at_vertex.items():
            # Add a vertex for each component in this levelset
            nextNodeName = R.get_next_vert_name()
            R.add_node(nextNodeName, now_min)
            vert_to_component[nextNodeName] = comp  # Store the component represented by this vertex
            verts_at_level.append(nextNodeName)
            
            # Check if any simplex in vertex component is a subset of any of simplices in a previous edge's component
            for e in edges_at_prev_level:
                prev_comp = vert_to_component[e]
                if any([is_face(prev_simp, simp) for simp in comp for prev_simp in prev_comp]):
                    R.add_edge(e, nextNodeName)
        
        #----
        # Add the edge vertices for after the vertex is passed  
        #----

        # Remove the vertex from the level set
        if [vert] in currentLevelSet:
            currentLevelSet.remove([vert])
            
        # Add the upper star to the current level set
        for s in upper_star:
            if s not in currentLevelSet:
                currentLevelSet.append(s)

        components = get_levelset_components(currentLevelSet)
        if verbose:
            print(f"\n  Updated current level set simplices: {currentLevelSet}")
            print(f"  Level set components after vertex {vert} (func val {filt:.2f}):")
            for comp in components.values():
                print(f"    Component: {comp}")
        #----
        # Set up a vertex in the Reeb graph for each connected component
        # These will represent edges
        # These are at height (now_min + now_max)/2 
        #----
        edges_at_prev_level = []
        for comp in components.values():
            # Create a new vertex in the Reeb graph
            e_name = 'e_'+str(half_edge_index)
            R.add_node(e_name, (now_min + now_max) / 2)
            vert_to_component[e_name] = comp  # Store the component represented by this half edge top
            half_edge_index += 1
            edges_at_prev_level.append(e_name)
            
            # Now connect to the vertices at this level 
            for v in verts_at_level:
                
                # Get the component represented by vertex v
                prev_comp = vert_to_component[v]

                if any([is_face(simp, prev_simp) for simp in comp for prev_simp in prev_comp]):
                    R.add_edge(v, e_name)

    return R
    