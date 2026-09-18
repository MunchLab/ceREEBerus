import random
import unittest

from cereeberus.compute.computereeb import get_levelset_components
from cereeberus.compute.unionfind import UnionFind


def _get_levelset_components_reference(L):
    """Reference O(n^2) implementation kept here only to check the fast
    version against it; this is the pairwise-is_face algorithm that used
    to live in get_levelset_components before the O(n) fix (issue: this
    function was quadratic via pairwise is_face checks)."""

    def is_face(sigma, tau):
        return set(tau).issubset(set(sigma))

    UF = UnionFind(range(len(L)))
    for i, simplex1 in enumerate(L):
        for j, simplex2 in enumerate(L):
            if i < j:
                if is_face(simplex1, simplex2) or is_face(simplex2, simplex1):
                    UF.union(i, j)

    components_index = UF.components_dict()
    return {tuple(L[k]): [L[i] for i in v] for k, v in components_index.items()}


def _normalize(components):
    return sorted(
        sorted(tuple(sorted(s)) for s in comp) for comp in components.values()
    )


class TestGetLevelsetComponents(unittest.TestCase):

    def test_matches_reference_on_randomized_inputs(self):
        # Regression test for the O(n^2) -> O(n) rewrite: the fast
        # dict-based version must agree with the original pairwise
        # is_face implementation on every randomized level set.
        random.seed(0)
        for _ in range(500):
            L = [
                sorted(random.sample(range(10), random.randint(1, 3)))
                for _ in range(15)
            ]
            fast = get_levelset_components(L)
            reference = _get_levelset_components_reference(L)
            self.assertEqual(_normalize(fast), _normalize(reference))

    def test_matches_reference_with_tetrahedra(self):
        # Same check, but allowing 4-vertex simplices too, to confirm the
        # general subset enumeration (not just a triangle-mesh special
        # case) matches the original pairwise is_face algorithm.
        random.seed(1)
        for _ in range(300):
            L = [
                sorted(random.sample(range(8), random.randint(1, 4)))
                for _ in range(12)
            ]
            fast = get_levelset_components(L)
            reference = _get_levelset_components_reference(L)
            self.assertEqual(_normalize(fast), _normalize(reference))

    def test_isolated_vertices_are_singleton_components(self):
        L = [[0], [5], [9]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 3)

    def test_vertex_is_face_of_edge(self):
        L = [[0], [1], [0, 1]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 1)

    def test_vertex_is_face_of_triangle(self):
        L = [[2], [0, 1, 2]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 1)

    def test_edge_is_face_of_triangle(self):
        L = [[0, 1], [0, 1, 2]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 1)

    def test_disjoint_simplices_stay_separate(self):
        L = [[0, 1], [2, 3]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 2)

    def test_exact_duplicate_simplices_merge(self):
        L = [[0, 1], [1, 0]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 1)

    def test_empty_levelset(self):
        self.assertEqual(get_levelset_components([]), {})

    def test_tetrahedron_and_its_triangular_face(self):
        # A 4-vertex simplex (tetrahedron) and a 3-vertex simplex that is
        # exactly one of its faces should land in the same component.
        # This is the case a naive triangle-only face enumeration misses.
        L = [[0, 1, 2, 3], [1, 2, 3]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 1)

    def test_tetrahedron_and_its_edge(self):
        L = [[0, 1, 2, 3], [0, 1]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 1)

    def test_disjoint_tetrahedra(self):
        L = [[0, 1, 2, 3], [4, 5, 6, 7]]
        components = get_levelset_components(L)
        self.assertEqual(len(components), 2)


if __name__ == "__main__":
    unittest.main()