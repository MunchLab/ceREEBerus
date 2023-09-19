
#=================================
# Construct an example Reeb
# graph of a simple upright torus
#=================================

from cereeberus.reeb.graph import Reeb
from cereeberus.data import graphs


def torus():
    '''
    Returns the Reeb graph of a simple upright torus as a Reeb class. 
    '''
    return Reeb(graphs.torus_graph())

def dancing_man():
    '''
    Returns the Reeb graph of the dancing man as a Reeb class. 
    '''
    return Reeb(graphs.dancing_man())

def juggling_man():
    '''
    Returns the Reeb graph of the juggling man as a Reeb class. 
    '''
    return Reeb(graphs.juggling_man())

def simple_loops():
    '''
    Returns the Reeb graph of the simple loops example
    '''
    return Reeb(graphs.simple_loops())

def simple_loops_unordered():
    '''
    Returns the Reeb graph of the simple loops example
    '''
    return Reeb(graphs.simple_loops_unordered())
