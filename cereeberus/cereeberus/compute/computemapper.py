from ..reeb.mapper import MapperGraph


# Interprets the lensfunction as a python function, does it, then returns the new location of each point for every point
def __runlensfunction(lensfunction, pointcloud):
    lensfunctionoutput = []
    if callable(lensfunction):
        for val in range(len(pointcloud)):
            lensfunctionoutput.append(
                [lensfunction(pointcloud[val]), tuple(pointcloud[val])]
            )
        return lensfunctionoutput

    if len(lensfunction) != len(pointcloud):
        raise ValueError(
            "If the lens function is given as a list of numbers, it must have the same length as the number of points in the point cloud."
        )

    for val in range(len(pointcloud)):
        lensfunctionoutput.append([lensfunction[val], tuple(pointcloud[val])])

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
def __cluster(coveringsets, clusteralgorithm):
    # trivial clustering
    if clusteralgorithm == "trivial":
        finished_cluster = list()
        cluster = list()
        for val1 in range(len(coveringsets)):
            cluster.append(coveringsets[val1][0][2])
            for val2 in range(1, len(coveringsets[val1])):
                cluster.append(
                    (coveringsets[val1][val2][1][0], coveringsets[val1][val2][1][1])
                )
            finished_cluster.append(
                cluster[:]
            )  # Works like this to avoid passing by reference in python lists
            cluster.clear()
        # print("Clustering Output: ")
        # print(finished_cluster)
        return finished_cluster
    # execute sklearn clusterings
    elif callable(clusteralgorithm):
        finished_cluster = list()
        coverpointcloud = list()
        cluster = list()
        for val1 in range(len(coveringsets)):
            # alters data to fit with sklearn clustering algorithms
            for val2 in range(1, len(coveringsets[val1])):
                coverpointcloud.append(
                    (coveringsets[val1][val2][1][0], coveringsets[val1][val2][1][1])
                )
            # does clustering algorithm
            cluster_out = clusteralgorithm(coverpointcloud)
            # puts points into list
            for val2 in range(max(cluster_out.labels_) + 1):
                cluster.append(
                    [coveringsets[val1][0][2]]
                )  # The position of the covering set in the cover (preserved for distance purposes)
                for val3 in range(len(cluster_out.labels_)):
                    if cluster_out.labels_[val3] == val2:
                        cluster[val2].append(coverpointcloud[val3])
                finished_cluster.append(cluster[val2])
            coverpointcloud.clear()
            cluster.clear()
        # print("Clustering Output: ")
        # print(finished_cluster)
        return finished_cluster
    else:
        print("input not valid")
        return list()


# Adds edges between the cluster that share points
def __addedges(clusterpoints):
    outputgraph = MapperGraph()
    val2 = 0
    for val1 in range(len(clusterpoints)):
        outputgraph.add_node(val1, clusterpoints[val1][0])
        while val2 < val1:
            if clusterpoints[val1][0] != clusterpoints[val2][0]:
                if len(set(clusterpoints[val1]) & set(clusterpoints[val2])) > 0:
                    outputgraph.add_edge(val1, val2)
            val2 += 1
        val2 = 0
    # print("Final Output: ")
    # print(outputgraph)
    return outputgraph


# Does the Mapper Algorithm in order
def computeMapper(pointcloud, lensfunction, cover, clusteralgorithm):
    """
    Computes the mapper graph of an input function. 
    
    The point cloud should be given as a list of tuples or as a numpy array. 
    
    The lens function should be given as either a list of numbers with the same length as the number of points; or as a callable function where :math:`f(point) = \text{value}` so long as the function can be determined from the coordinate values of the point.    
    
    The cover should be given as a list of intervals. This can be done, for example, using the 'cereeberus.cover' function in this module, which takes in a minimum, maximum, number of covers, and percentage of overlap to create a cover. 
    
    The clustering algorithm should be given as a callable that takes in a point cloud and outputs cluster labels (for example, `sklearn.cluster.DBSCAN(min_samples=2,eps=0.3).fit`).

    Parameters:
        A pointcloud (as a list of tuples or as a numpy array)
        A lens function (as a callable or a list of numbers)
        A cover (as a list of intervals)
        A clustering algorithm (as a callable)

    Returns:
        A `MapperGraph` object representing the mapper graph of the input data and lens function.
    """
    
    lensfunctionoutput = __runlensfunction(lensfunction, pointcloud)                           
    coveringsets = __createcoveringsets(lensfunctionoutput, cover)
    clusterpoints = __cluster(coveringsets, clusteralgorithm)
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
