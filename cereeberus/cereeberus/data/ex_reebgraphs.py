from cereeberus import ReebGraph
from cereeberus.data import ex_graphs


def line(a = 0, b = 1, seed=None):
    """
    Returns the Reeb graph of a simple line as a ReebGraph class. The endpoints have function value a and b respectively.

    Parameters:
        a, b (int): The function values for the two vertices in increasing order.
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        ReebGraph: The Reeb graph of the line.

        
    .. figure:: ../../images/line.png
        :figwidth: 400px

    """

    R = ReebGraph()
    R.add_node('a', a)
    R.add_node('b', b)
    R.add_edge('a', 'b')
    R.set_pos_from_f(seed=seed)

    return R

def torus(a = 0, b = 1, c = 4, d = 5, multigraph = True, seed=None):
    '''
    Returns the Reeb graph of a simple upright torus as a ReebGraph class. 

    Parameters:
        a ,b, c, d (int): The function values for the four vertices in increasing order.
        multigraph (bool): Optional. If False, then the loop edges will be subdivided so that the resulting graph doesn't have multiple edges between $b$ and $c$. Default is True.
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.


    Returns:
        ReebGraph: The Reeb graph of the torus.
    
    .. figure:: ../../images/torus.png
        :figwidth: 400px

    '''
    R = ReebGraph()
    R.add_node('a', a)
    R.add_node('b', b)
    R.add_node('c', c)
    R.add_node('d', d)
    R.add_edge('a', 'b')
    R.add_edge('b', 'c')
    R.add_edge('b', 'c')
    R.add_edge('c', 'd')

    if not multigraph:
        R.subdivide_edge('b', 'c', 'e', (b+c)/2) 
        R.subdivide_edge('b', 'c', 'f', (b+c)/2) 

    R.set_pos_from_f(seed=seed)

    return R

def dancing_man(seed=None):
    '''
    Returns the Reeb graph of the dancing man as a ReebGraph class. 

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        ReebGraph: The Reeb graph of the dancing man.
    
    .. figure:: ../../images/dancing_man.png
        :figwidth: 400px
    '''
    return ReebGraph(ex_graphs.dancing_man(), seed=seed)

def juggling_man(seed=None):
    '''
    Returns the Reeb graph of the juggling man as a ReebGraph class. 

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        ReebGraph: The Reeb graph of the juggling man.

    .. figure:: ../../images/juggling_man.png
        :figwidth: 400px
    '''
    return ReebGraph(ex_graphs.juggling_man(), seed=seed)

def simple_loops(seed=None):
    '''
    Returns the Reeb graph of the simple loops example.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        ReebGraph: The Reeb graph of the simple loops example.

    .. figure:: ../../images/simple_loops.png
        :figwidth: 400px
    '''
    return ReebGraph(ex_graphs.simple_loops(), seed=seed)

def simple_loops_unordered(seed=None):
    '''
    Returns the Reeb graph of the simple loops example.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        ReebGraph: The Reeb graph of the simple loops example.
    '''
    return ReebGraph(ex_graphs.simple_loops_unordered(), seed=seed)

def interleave_example_A(seed=None):
    '''
    Returns the Reeb graph of the first example for the interleave function.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        ReebGraph: The Reeb graph of the first example for the interleave function.

    .. figure:: ../../images/interleave_example_A.png
        :figwidth: 400px
    '''

    R = dancing_man()
    R.subdivide_edge(6,7,8, 3)
    R.add_edge(8,1)
    R.f = {v: 2*R.f[v] for v in R.nodes()}
    R.set_pos_from_f(seed=seed)


    return R

def interleave_example_B(seed=None):
    '''
    Returns the Reeb graph of the second example for the interleave function.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.
    
    Returns:
        ReebGraph: The Reeb graph of the second example for the interleave function.
    
    .. figure:: ../../images/interleave_example_B.png
        :figwidth: 400px
    '''
    R = interleave_example_A()
    R.f[8] = 3
    R.f[5] = 3
    R.remove_edge(3,1)
    R.set_pos_from_f(seed = seed)
    return R