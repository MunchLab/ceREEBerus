from cereeberus import ReebGraph, MapperGraph
from cereeberus.data import ex_graphs

def torus(delta = .1, seed=None):
    '''
    Returns the Mapper graph of a simple upright torus as a MapperGraph class. 

    Parameters:
        delta (float): Optional. The delta value to use for the Mapper graph.
        seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

    Returns:
        Mapper: The mapper graph of the torus.

    
    .. figure:: ../../images/torus_mapper.png
        :figwidth: 400px

    '''
    return ReebGraph(ex_graphs.torus_graph(), seed=seed).to_mapper(delta = delta)

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
