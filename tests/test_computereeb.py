import math
import unittest

import numpy as np

from cereeberus.reeb.lowerstar import LowerStar
from cereeberus.compute.computereeb import computeReeb, _merge_close_values


class TestMergeCloseValues(unittest.TestCase):
    """Unit tests for the _merge_close_values helper directly."""

    def test_ulp_adjacent_values_are_merged(self):
        H = 1.0
        H2 = math.nextafter(H, math.inf)  # 1 ULP above H
        out = _merge_close_values([0.0, H, H2, 5.0])
        self.assertEqual(out[1], out[2])  # H and H2 snapped to the same value

    def test_genuinely_distinct_values_are_not_merged(self):
        vals = [0.0, 1.0, 2.0, 3.0]
        out = _merge_close_values(vals)
        self.assertEqual(out, vals)

    def test_chain_of_three_close_values_all_merge_to_first(self):
        H = 1.0
        H2 = math.nextafter(H, math.inf)
        H3 = math.nextafter(H2, math.inf)
        out = _merge_close_values([H, H2, H3])
        self.assertEqual(out, [H, H, H])

    def test_values_far_apart_by_absolute_gap_not_merged(self):
        # rtol alone would flag these as close at large magnitude; atol/rtol
        # combination here should still treat a real gap as real.
        out = _merge_close_values([1000.0, 1000.1])
        self.assertEqual(out, [1000.0, 1000.1])


class TestComputeReebTiedHeights(unittest.TestCase):
    """Regression tests for the ULP-adjacent tied-heights crash.

   computeReeb raised 'ValueError: The vertex X must be in the Reeb graph...' when two
    distinct filtration values were 1 float64 ULP apart, causing the
    half-edge sentinel midpoint (now_min + now_max) / 2 to round to
    exactly one of them and spuriously trigger the tie-collapse branch.
    """

    def _build_ulp_adjacent_case(self):
        """Minimal 5-vertex hand-traceable repro of the ULP-collision bug.

        Two triangles sharing an edge (2,3); vertices 2 and 3 sit 1 ULP
        apart in height, close enough that the algorithm's own midpoint
        calculation could not previously tell them apart from a sentinel.
        """
        H = 1.0
        H2 = math.nextafter(H, math.inf)  # 1 ULP above H
        K = LowerStar()
        K.insert([0, 2, 3])
        K.insert([1, 2, 3])
        K.insert([2, 3, 4])
        K.assign_filtration(0, 0.0)
        K.assign_filtration(1, 0.0)
        K.assign_filtration(2, H)
        K.assign_filtration(3, H2)
        K.assign_filtration(4, H + 1.0)
        return K

    def test_ulp_adjacent_heights_do_not_crash(self):
        K = self._build_ulp_adjacent_case()
        try:
            R = computeReeb(K)
        except ValueError as e:
            self.fail(f"computeReeb raised ValueError on ULP-adjacent heights: {e}")

        # Structural sanity: should be a connected tree (no cycles introduced).
        self.assertEqual(R.number_of_edges(), R.number_of_nodes() - 1)

    def test_exact_tie_still_works(self):
        """Regression guard: genuine exact ties must still collapse correctly
        (this is the originally-fixed, closed issue -- must not regress)."""
        K = LowerStar()
        K.insert([0, 2, 3])
        K.insert([1, 2, 3])
        K.insert([2, 3, 4])
        K.assign_filtration(0, 0.0)
        K.assign_filtration(1, 0.0)
        K.assign_filtration(2, 1.0)
        K.assign_filtration(3, 1.0)  # exact tie with vertex 2
        K.assign_filtration(4, 2.0)

        R = computeReeb(K)
        self.assertEqual(R.number_of_edges(), R.number_of_nodes() - 1)

    def test_docstring_example_unchanged(self):
        """Non-degenerate case with no near-ties: output must be identical
        to pre-fix behavior."""
        K = LowerStar()
        K.insert([0, 1, 2])
        K.insert([1, 3])
        K.insert([2, 3])
        K.assign_filtration([0], 0.0)
        K.assign_filtration([1], 3.0)
        K.assign_filtration([2], 5.0)
        K.assign_filtration([3], 7)

        R = computeReeb(K)
        self.assertEqual(R.number_of_nodes(), 10)
        self.assertEqual(R.number_of_edges(), 10)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("trimesh") is not None,
    "trimesh not installed; skipping real-mesh regression test",
)
class TestComputeReebTiedHeightsRealMesh(unittest.TestCase):
    """Real-world confirmation on a symmetric mesh (optional; requires trimesh).

    Icospheres reliably produce vertices whose true height is identical by
    construction but that land 1 ULP apart due to differing subdivision
    paths.
    """

    def test_icosphere_subdivisions_2_does_not_crash(self):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2)
        K = LowerStar()
        for tri in mesh.faces:
            K.insert([int(v) for v in tri])
        for i, v in enumerate(mesh.vertices):
            K.assign_filtration(i, float(v[2]))

        R = computeReeb(K, verbose=False)
        self.assertEqual(R.number_of_nodes(), 81)
        self.assertEqual(R.number_of_edges(), 80)
        self.assertEqual(R.number_of_edges(), R.number_of_nodes() - 1)


if __name__ == "__main__":
    unittest.main()