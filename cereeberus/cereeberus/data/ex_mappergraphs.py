from cereeberus import ReebGraph, MapperGraph
from cereeberus.data import ex_graphs
from cereeberus.data import ex_reebgraphs as ex_rg


def line(a = 0, b = 3, seed = None):
    """
    Returns the Mapper graph of a simple line as a MapperGraph class. The endpoints have function value a and b respectively.

    Parameters:
        a, b (int): The function values for the two vertices in increasing order.
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        MapperGraph: The Mapper graph of the line.

        
    .. figure:: ../../images/line_mapper.png
        :figwidth: 400px

    """

    MG = ex_rg.line(a, b, seed = seed).to_mapper()
    MG.set_pos_from_f(seed = seed)

    return MG

def torus(a = 0, b = 1, c = 4, d = 5, 
            delta = .1, seed=None):
    '''
    Returns the Mapper graph of a simple upright torus as a MapperGraph class. 

    Parameters:
        a ,b, c, d (int): The integer function values for the four vertices in increasing order.
        delta (float): Optional. The delta value to use for the Mapper graph.
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        Mapper: The mapper graph of the torus.

    
    .. figure:: ../../images/torus_mapper.png
        :figwidth: 400px

    '''
    # make sure a, b, c, and d are integers
    if not all(isinstance(x, int) for x in [a, b, c, d]):
        raise ValueError('a, b, c, and d must be integers.')
    MG = ex_rg.torus(a,b,c,d, multigraph = True, seed = seed).to_mapper(delta = delta)
    MG.set_pos_from_f(seed = seed)
    return MG

def dancing_man(delta = .1, seed=None):
    '''
    Returns the Mapper graph of the dancing man as a MapperGraph class. 

    Parameters:

        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        MapperGraph: The mapper graph of the dancing man.
    
    .. figure:: ../../images/dancing_man_mapper.png
        :figwidth: 400px

    '''
    return ReebGraph(ex_graphs.dancing_man(), seed=seed).to_mapper(delta = delta)

def juggling_man(delta = .1, seed=None):
    '''
    Returns a modified mapper graph of the juggling man as a MapperGraph class. Some vertex locations were moved to make them integers. 

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        MapperGraph: The mapper graph of the juggling man.
    
    .. figure:: ../../images/juggling_man_mapper.png
        :figwidth: 400px

    '''
    R = ReebGraph(ex_graphs.juggling_man(), seed=seed)
    R.f[9] = 7
    R.f[8] = 6
    R.f[10] = 5

    return R.to_mapper(delta = delta)

def simple_loops(delta = .1, seed=None):
    '''
    Returns the mapper graph of the simple loops example.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        MapperGraph: The mapper graph of the simple loops example.
    
    .. figure:: ../../images/simple_loops_mapper.png
        :figwidth: 400px

    '''
    return ReebGraph(ex_graphs.simple_loops(), seed=seed).to_mapper()

def simple_loops_unordered(seed=None):
    '''
    Returns the mapper graph of the unordered simple loops example.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        MapperGraph: The mapper graph of the simple loops example.

    '''
    return ReebGraph(ex_graphs.simple_loops_unordered(), seed=seed).to_mapper()

def interleave_example_A(seed = None):
    '''
    Returns the mapper graph of the first example for the interleave function.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        MapperGraph: The mapper graph of the first example for the interleave function.
    
    .. figure:: ../../images/interleave_example_A_mapper.png
        :figwidth: 400px

    '''
    return ex_rg.interleave_example_A(seed = seed).to_mapper()

def interleave_example_B(seed = None):
    '''
    Returns the mapper graph of the second example for the interleave function.

    Parameters:
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        MapperGraph: The mapper graph of the second example for the interleave function.
    
    .. figure:: ../../images/interleave_example_B_mapper.png
        :figwidth: 400px

    '''
    return ex_rg.interleave_example_B(seed = seed).to_mapper()