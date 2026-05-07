from ..reeb.mapper import MapperGraph
import numpy as np


# Interprets the lensfunction as a python function, does it, then returns the new location of each point for every point.
# Point identifiers are always integer indices.
def __runlensfunction(lensfunction, pointcloud, distance_matrix=None):
    n = len(distance_matrix) if pointcloud is None else len(pointcloud)
    lensfunctionoutput = []

    if callable(lensfunction):
        if pointcloud is None:
            raise ValueError(
                "A callable lens function requires a pointcloud. "
                "When using only a distance_matrix, provide lensfunction as a precomputed list."
            )
        for i in range(n):
            lensfunctionoutput.append([lensfunction(pointcloud[i]), i])
        return lensfunctionoutput

    if len(lensfunction) != n:
        raise ValueError(
            "If the lens function is given as a list of numbers, it must have the same length as the number of points."
        )

    for i in range(n):
        lensfunctionoutput.append([lensfunction[i], i])

    return lensfunctionoutput


# Creates a list of covers, together with any points inside the cover, ie, [[cover, point1, point2], [cover, point3]]
# Also removes any covers that have no points inside of them
# this would probably be better done as a struct, containing a covering set, the covering set's position in the cover, and an array of points
def __createcoveringsets(points, cover):
    # adds location information to cover
    for val0 in range(len(cover)):
        cover[val0] = (cover[val0][0], cover[val0][1], val0)
    # creates the list
    coveringsets = []
    for val1 in range(len(cover)):
        coveringsets.append([cover[val1]])
        for val2 in range(len(points)):
            if points[val2][0] >= cover[val1][0] and points[val2][0] <= cover[val1][1]:
                coveringsets[val1].append(points[val2])
    # removes unused covers
    position = 0
    while position < len(coveringsets):
        if len(coveringsets[position]) == 1:
            coveringsets.pop(position)
        else:
            position += 1
    # print("Covering Sets Output: ")
    # print(coveringsets)
    return coveringsets


# cluster the points using a number of existing clustering algorithms
def __cluster(coveringsets, clusteralgorithm, pointcloud=None, distance_matrix=None):
    # trivial clustering
    if clusteralgorithm == "trivial":
        finished_cluster = list()
        for val1 in range(len(coveringsets)):
            cluster = [coveringsets[val1][0][2]]
            for val2 in range(1, len(coveringsets[val1])):
                cluster.append(coveringsets[val1][val2][1])
            finished_cluster.append(cluster)
        return finished_cluster
    # execute sklearn clusterings
    elif callable(clusteralgorithm):
        finished_cluster = list()
        for val1 in range(len(coveringsets)):
            indices = [coveringsets[val1][val2][1] for val2 in range(1, len(coveringsets[val1]))]
            if distance_matrix is not None:
                sub_matrix = distance_matrix[np.ix_(indices, indices)]
                cluster_out = clusteralgorithm(sub_matrix)
            else:
                coverpointcloud = [pointcloud[idx] for idx in indices]
                cluster_out = clusteralgorithm(coverpointcloud)
            cluster = list()
            for val2 in range(max(cluster_out.labels_) + 1):
                cluster.append([coveringsets[val1][0][2]])
                for val3 in range(len(cluster_out.labels_)):
                    if cluster_out.labels_[val3] == val2:
                        cluster[val2].append(indices[val3])
                finished_cluster.append(cluster[val2])
        return finished_cluster
    else:
        print("input not valid")
        return list()


# Adds edges between the cluster that share points
def __addedges(clusterpoints):
    outputgraph = MapperGraph()
    for val1 in range(len(clusterpoints)):
        outputgraph.add_node(val1, clusterpoints[val1][0])
        for val2 in range(val1):
            if clusterpoints[val1][0] == clusterpoints[val2][0]:
                continue
            # Compare only point memberships (skip cover index at position 0).
            if len(set(clusterpoints[val1][1:]) & set(clusterpoints[val2][1:])) > 0:
                outputgraph.add_edge(val1, val2)
    # print("Final Output: ")
    # print(outputgraph)
    return outputgraph


# Does the Mapper Algorithm in order
def computeMapper(pointcloud, lensfunction, cover, clusteralgorithm, distance_matrix=None):
    """
    Computes the mapper graph of an input function.

    The point cloud should be given as a list of tuples or as a numpy array. It may be
    ``None`` when a ``distance_matrix`` is provided and the lens function is a precomputed list.

    The lens function should be given as either a list of numbers with the same length as the
    number of points; or as a callable function where :math:`f(point) = \\text{value}` so long
    as the function can be determined from the coordinate values of the point. When only a
    ``distance_matrix`` is supplied (no ``pointcloud``), the lens function must be a precomputed
    list.

    The cover should be given as a list of intervals. This can be done, for example, using the
    ``cereeberus.cover`` function in this module.

    The clustering algorithm should be given as a callable that takes in a point cloud and
    outputs cluster labels (for example, ``sklearn.cluster.DBSCAN(min_samples=2, eps=0.3).fit``).
    When a ``distance_matrix`` is provided, the callable receives a precomputed square
    sub-matrix for each cover set, so it should be configured with ``metric='precomputed'``
    (for example, ``sklearn.cluster.DBSCAN(metric='precomputed', eps=0.3).fit``).

    Parameters:
        pointcloud: a list of tuples or numpy array of coordinates, or ``None`` when using a distance matrix without coordinates.
        lensfunction: a callable ``f(point) -> value`` or a precomputed list of values.
        cover: a list of intervals (see ``cereeberus.cover``).
        clusteralgorithm: ``'trivial'`` or a callable clustering algorithm.
        distance_matrix: optional square numpy array of pairwise distances. When provided, clustering uses this matrix rather than raw coordinates.

    Returns:
        A ``MapperGraph`` object representing the mapper graph of the input data and lens function.
    """
    if pointcloud is None and distance_matrix is None:
        raise ValueError("Either pointcloud or distance_matrix must be provided.")

    if distance_matrix is not None:
        distance_matrix = np.asarray(distance_matrix)
        n = len(pointcloud) if pointcloud is not None else len(distance_matrix)
        if distance_matrix.shape != (n, n):
            raise ValueError(
                "distance_matrix must be a square array with one row/column per point."
            )

    lensfunctionoutput = __runlensfunction(lensfunction, pointcloud, distance_matrix)
    coveringsets = __createcoveringsets(lensfunctionoutput, cover)
    clusterpoints = __cluster(coveringsets, clusteralgorithm, pointcloud, distance_matrix)
    outputgraph = __addedges(clusterpoints)

    return outputgraph


# function to create covers
# cover(min, max, #covers, %overlap)
def cover(min=-1, max=1, numcovers=10, percentoverlap=0.5):
    """
    Creates a cover to be used for inputs in the computeMapper function

    Parameters:
        min: the minimum for the range of the covering sets
        max: the maximum for the range of the covering sets
        numcovers: number of covers to create
        percentoverlap: percentage (from 0 to 1) of overlap between covers

    Returns:
        An array of intervals
    """
    output = []
    val = 0
    coversize = (max - min) / numcovers * (1 + (percentoverlap))
    while val < numcovers:
        center = (min * (numcovers - (val + 0.5)) + max * (val + 0.5)) / numcovers
        output.append(((-0.5 * coversize) + center, (0.5 * coversize) + center))
        val += 1
    return output
