#This is probably bad
from asteval import Interpreter
aeval = Interpreter()

from ..reeb.mapper import MapperGraph





#Interprets the lensfunction as a python function, does it, then returns the new location of each point for every point
def __runlensfunction(lensfunction, pointcloud):
    lensfunctionoutput = []
    for val in range(len(pointcloud)):
        aeval.symtable['x'] = pointcloud[val][0]
        aeval.symtable['y'] = pointcloud[val][1]
        lensfunctionoutput.append((aeval(lensfunction), pointcloud[val][0], pointcloud[val][1]))
    return lensfunctionoutput


#Creates a list of covers, together with any points inside the cover, ie, [[cover, point1, point2], [cover, point3]]
#Also removes any covers that have no points inside of them
#this would probably be better done as a struct, containing a covering set, the covering set's position in the cover, and an array of points
def __createcoveringsets(points, cover):
    #adds location information to cover
    for val0 in range(len(cover)):
        cover[val0] = (cover[val0][0], cover[val0][1], val0)
    #creates the list
    coveringsets = []
    for val1 in range(len(cover)):
        coveringsets.append([cover[val1]])
        for val2 in range(len(points)):
            if points[val2][0] >= cover[val1][0] and points[val2][0] <= cover[val1][1]:
                coveringsets[val1].append(points[val2])
    #removes unused covers
    position = 0
    while position < len(coveringsets):
        if len(coveringsets[position]) == 1:
            coveringsets.pop(position)
        else:
            position += 1
    return coveringsets


#cluster the points using a number of existing clustering algorithms
def __cluster(coveringsets, clusteralgorithm):
    #trivial clustering
    if clusteralgorithm == "trivial":
        for val in range(len(coveringsets)):
            location = coveringsets[val].pop(0)
            coveringsets[val].insert(0, location[2])
        return coveringsets
    #execute sklearn clusterings
    elif callable(clusteralgorithm):
        finished_cluster = list()
        coverpointcloud = list()
        cluster = list()
        for val1 in range(len(coveringsets)):
            #alters data to fit with sklearn clustering algorithms
            for val2 in range(1, len(coveringsets[val1])):   
                coverpointcloud.append((coveringsets[val1][val2][1],coveringsets[val1][val2][2]))
            #does clustering algorithm
            cluster_out = clusteralgorithm(coverpointcloud)
            #puts points into list
            for val2 in range(max(cluster_out.labels_)+1):
                cluster.append([coveringsets[val1][0][2]])   #The position of the covering set in the cover (preserved for distance purposes)
                for val3 in range(len(cluster_out.labels_)):
                    if cluster_out.labels_[val3] == val2:
                        cluster[val2].append(coverpointcloud[val3])
                finished_cluster.append(cluster[val2])
            coverpointcloud.clear()
            cluster.clear()
        return finished_cluster
    else:
        print("input not valid")
        return list()


#Adds edges between the cluster that share points
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
    return outputgraph


#Does the Mapper Algorithm in order
def runmapper(self, pointcloud, lensfunction, cover, clusteralgorithm):
    """
    Computes the Mapper Alogirthm

    Parameters:
        A pointcloud (as a list)
        A lens function (as a string)
        A cover (as a list of intervals)
        A clustering algorithm (as a callable)

    Returns:
        A MapperGraph object as given by the Mapper Algorithm run on the parameters
    """
    lensfunctionoutput = self.__runlensfunction(lensfunction, pointcloud) #move to compute folder, change name to computemapper
    coveringsets = self.__createcoveringsets(lensfunctionoutput, cover)
    clusterpoints = self.__cluster(coveringsets, clusteralgorithm)
    outputgraph = self.__addedges(clusterpoints)
    return outputgraph


#function to create covers
#cover(min, max, #covers, %overlap)
def cover(min=-1, max=1, numcovers=10, percentoverlap=.5):
    """
    Creates a cover to be used for inputs in the runmapper function

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
    coversize = (max - min)/numcovers * (1+(percentoverlap))
    while val < numcovers:
        center = (min*(numcovers-(val+0.5)) + max*(val+0.5))/numcovers
        output.append(((-0.5*coversize) + center, (0.5*coversize) + center))
        val += 1
    return output
    




