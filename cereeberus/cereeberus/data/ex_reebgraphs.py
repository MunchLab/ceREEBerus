from cereeberus import ReebGraph
from cereeberus.data import ex_graphs

def torus(seed=None):
    def torus(seed=None):
        '''
        Returns the Reeb graph of a simple upright torus as a ReebGraph class. 

        Parameters:
            seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

        Returns:
            ReebGraph: The Reeb graph of the torus.
        
        .. figure:: images/torus.png
            :figwidth: 200px

        '''
        return ReebGraph(ex_graphs.torus_graph(), seed=seed)

    def dancing_man(seed=None):
        '''
        Returns the Reeb graph of the dancing man as a ReebGraph class. 

        Parameters:
            seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

        Returns:
            ReebGraph: The Reeb graph of the dancing man.
        '''
        return ReebGraph(ex_graphs.dancing_man(), seed=seed)

    def juggling_man(seed=None):
        '''
        Returns the Reeb graph of the juggling man as a ReebGraph class. 

        Parameters:
            seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

        Returns:
            ReebGraph: The Reeb graph of the juggling man.
        '''
        return ReebGraph(ex_graphs.juggling_man(), seed=seed)

    def simple_loops(seed=None):
        '''
        Returns the Reeb graph of the simple loops example.

        Parameters:
            seed (int): Optional. The seed to use for the random number generator, which only controls the layout function.

        Returns:
            ReebGraph: The Reeb graph of the simple loops example.
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
    return ReebGraph(ex_graphs.torus_graph(), seed=seed)

def dancing_man(seed=None):
    '''
    Returns the Reeb graph of the dancing man as a ReebGraph class. 
    '''
    return ReebGraph(ex_graphs.dancing_man(), seed=seed)

def juggling_man(seed=None):
    '''
    Returns the Reeb graph of the juggling man as a ReebGraph class. 
    '''
    return ReebGraph(ex_graphs.juggling_man(), seed=seed)

def simple_loops(seed=None):
    '''
    Returns the Reeb graph of the simple loops example
    '''
    return ReebGraph(ex_graphs.simple_loops(), seed=seed)

def simple_loops_unordered(seed=None):
    '''
    Returns the Reeb graph of the simple loops example
    '''
    return ReebGraph(ex_graphs.simple_loops_unordered(), seed=seed)
