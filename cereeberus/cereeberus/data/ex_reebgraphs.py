from cereeberus import ReebGraph
from cereeberus.data import ex_graphs

def torus(seed=None):
    '''
    Returns the Reeb graph of a simple upright torus as a ReebGraph class. 

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        ReebGraph: The Reeb graph of the torus.
    
    .. figure:: ../../images/torus.png
        :figwidth: 400px

    '''
    return ReebGraph(ex_graphs.torus_graph(), seed=seed)

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