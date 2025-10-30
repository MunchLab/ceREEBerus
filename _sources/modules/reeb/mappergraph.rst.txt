MapperGraph class
=================

This is the mapper graph class which is inherited from the Reeb graph class.  The main differences are 

1. The Mapper graph is a graph with a function, but all function values are stored as integers. There is also a stored ``delta`` value which is can be used for scaling, so that all function values can be thought of as ``delta * f(v)``. 
2. All edges have adjacent integer values. This equivalently means that no integer has a point in the interior of an edge in its inverse image.


Note that this class can be imported as::

    import cereeberus
    R = cereeberus.MapperGraph()

.. automodule:: cereeberus.reeb.mapper
    :members:
