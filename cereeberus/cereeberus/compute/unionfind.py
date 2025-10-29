# Edited from https://yuminlee2.medium.com/union-find-algorithm-ffa9cd7d2dba

class UnionFind:
    """Union find data structure 
    """
    def __init__(self, vertices):
        self.parent = {vertex: vertex for vertex in vertices}
        self.size = {vertex: 1 for vertex in vertices}
        self.count = len(vertices)
    

    # Time: O(logn) | Space: O(1)
    def find(self, node):
        while node != self.parent[node]:
            # path compression
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node
    
    # Time: O(1) | Space: O(1)
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        # already in the same set
        if root1 == root2:
            return

        if self.size[root1] > self.size[root2]:
            self.parent[root2] = root1
            self.size[root1] += 1
        else:
            self.parent[root1] = root2
            self.size[root2] += 1
        
        self.count -= 1

    def components_dict(self):
        """Returns a dictionary mapping component representatives to lists of elements in that component.

        Returns:
            dict: A dictionary where keys are component representatives and values are lists of elements in that component.
        """
        components = {}
        for node in self.parent:
            root = self.find(node)
            if root not in components:
                components[root] = []
            components[root].append(node)
        return components


if __name__ == "__main__":
    edges = [
        [0, 2],
        [1, 4],
        [1, 5],
        [2, 3],
        [2, 7],
        [4, 8],
        [5, 8],
    ]
    numberOfElements = 9

    uf = UnionFind(list(range(9)))

    for node1, node2 in edges:
        uf.union(node1, node2)
    
    print("number of connected components", uf.count)

# output: ('number of connected components', 3)