"""Tests for the PuLP 3.x / 4.x compatibility helpers in cereeberus.distance.ilp.

PuLP 4.0 removed ``LpVariable.dicts``, ``PULP_CBC_CMD``, ``prob.status`` and
``pulp.LpStatus``. These tests exercise both code paths with mocks, so they
run the same way no matter which PuLP version is installed, plus one small
end-to-end solve against the installed PuLP.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pulp
from cereeberus.distance.ilp import (
    _new_var,
    _new_var_dict,
    _solve,
    select_pulp_solver,
)


class FakePulp4Problem:
    """Minimal stand-in for a PuLP 4 LpProblem (has ``add_variable``)."""

    def __init__(self):
        self.created = []

    def add_variable(self, name, cat="Continuous"):
        self.created.append((name, cat))
        return f"var:{name}"


class TestNewVar(unittest.TestCase):
    def test_pulp4_path_uses_add_variable(self):
        prob = FakePulp4Problem()
        var = _new_var(prob, "minmax_var", cat="Integer")
        self.assertEqual(var, "var:minmax_var")
        self.assertEqual(prob.created, [("minmax_var", "Integer")])

    def test_pulp3_path_uses_lpvariable(self):
        prob = object()  # no add_variable -> PuLP 3 behaviour
        with patch("cereeberus.distance.ilp.pulp.LpVariable", create=True) as mock_var:
            mock_var.return_value = "legacy-var"
            var = _new_var(prob, "aux_0", cat="Integer")
        self.assertEqual(var, "legacy-var")
        mock_var.assert_called_once_with("aux_0", cat="Integer")


class TestNewVarDict(unittest.TestCase):
    def test_pulp4_path_keeps_tuple_keys(self):
        prob = FakePulp4Problem()
        indices = ((a, b) for a in range(2) for b in range(3))
        out = _new_var_dict(prob, "phi_n0_1", indices, cat="Binary")

        # Same keys as LpVariable.dicts would give, so build_map_matrices
        # can keep indexing with (a, b).
        self.assertEqual(sorted(out), [(a, b) for a in range(2) for b in range(3)])
        self.assertEqual(out[(1, 2)], "var:phi_n0_1_1_2")
        self.assertTrue(all(cat == "Binary" for _, cat in prob.created))
        # Every variable gets a unique name.
        names = [n for n, _ in prob.created]
        self.assertEqual(len(names), len(set(names)))

    def test_pulp4_path_triple_and_scalar_keys(self):
        prob = FakePulp4Problem()
        out3 = _new_var_dict(prob, "z", [(0, 1, 2)], cat="Binary")
        self.assertEqual(out3, {(0, 1, 2): "var:z_0_1_2"})
        out1 = _new_var_dict(prob, "y", [0, 1], cat="Integer")
        self.assertEqual(out1, {0: "var:y_0", 1: "var:y_1"})

    def test_pulp4_path_empty_indices(self):
        prob = FakePulp4Problem()
        self.assertEqual(_new_var_dict(prob, "empty", [], cat="Binary"), {})
        self.assertEqual(prob.created, [])

    def test_pulp3_path_uses_lpvariable_dicts(self):
        prob = object()
        indices = [(0, 0), (0, 1)]
        with patch("cereeberus.distance.ilp.pulp.LpVariable", create=True) as mock_var:
            mock_var.dicts.return_value = {"legacy": True}
            out = _new_var_dict(prob, "phi", indices, cat="Binary")
        self.assertEqual(out, {"legacy": True})
        mock_var.dicts.assert_called_once_with("phi", indices, cat="Binary")


class TestSolve(unittest.TestCase):
    def _pulp4_prob(self, code, status_str):
        prob = MagicMock(spec=["solve"])
        prob.solve.return_value = SimpleNamespace(status=code, status_str=status_str)
        return prob

    def test_pulp4_optimal(self):
        prob = self._pulp4_prob(1, "Optimal")
        self.assertEqual(_solve(prob, "solver"), (1, "Optimal"))
        prob.solve.assert_called_once_with("solver")

    def test_pulp4_shared_codes_use_pulp3_strings(self):
        expected = {
            0: "Not Solved",
            1: "Optimal",
            -1: "Infeasible",
            -2: "Unbounded",
            -3: "Undefined",
        }
        pulp4_names = {
            0: "NotSolved",
            1: "Optimal",
            -1: "Infeasible",
            -2: "Unbounded",
            -3: "Undefined",
        }
        for code, legacy in expected.items():
            with self.subTest(code=code):
                prob = self._pulp4_prob(code, pulp4_names[code])
                self.assertEqual(_solve(prob, None), (code, legacy))

    def test_pulp4_new_codes_pass_through(self):
        prob = self._pulp4_prob(-4, "TimeLimit")
        self.assertEqual(_solve(prob, None), (-4, "TimeLimit"))

    def test_pulp3_path_reads_prob_status(self):
        prob = MagicMock(spec=["solve", "status"])
        prob.solve.return_value = 1  # PuLP 3 returns an int
        prob.status = 1
        with patch(
            "cereeberus.distance.ilp.pulp.LpStatus",
            {0: "Not Solved", 1: "Optimal"},
            create=True,
        ):
            self.assertEqual(_solve(prob, None), (1, "Optimal"))


class TestSelectSolverPulp4(unittest.TestCase):
    @patch("cereeberus.distance.ilp.shutil.which", return_value=None)
    def test_missing_pulp_cbc_cmd_is_skipped(self, _mock_which):
        # Simulate PuLP 4: no PULP_CBC_CMD attribute, HiGHS available.
        fake_pulp = MagicMock(
            spec=["listSolvers", "COIN_CMD", "HiGHS", "HiGHS_CMD", "GLPK_CMD",
                  "GUROBI", "GUROBI_CMD"]
        )
        fake_pulp.listSolvers.return_value = ["PULP_CBC_CMD", "HiGHS"]
        fake_pulp.HiGHS.return_value = "highs-solver"

        with patch.dict(os.environ, {}, clear=True):
            with patch("cereeberus.distance.ilp.pulp", fake_pulp):
                solver = select_pulp_solver()

        self.assertEqual(solver, "highs-solver")
        fake_pulp.HiGHS.assert_called_once_with(msg=0)

    @patch("cereeberus.distance.ilp.shutil.which", return_value=None)
    def test_coin_cmd_preferred_over_highs(self, _mock_which):
        fake_pulp = MagicMock(spec=["listSolvers", "COIN_CMD", "HiGHS"])
        fake_pulp.listSolvers.return_value = ["HiGHS", "COIN_CMD"]
        fake_pulp.COIN_CMD.return_value = "coin-solver"

        with patch.dict(os.environ, {}, clear=True):
            with patch("cereeberus.distance.ilp.pulp", fake_pulp):
                solver = select_pulp_solver()

        self.assertEqual(solver, "coin-solver")
        fake_pulp.HiGHS.assert_not_called()

    @patch("cereeberus.distance.ilp.pulp.HiGHS", create=True)
    def test_highs_by_name(self, mock_highs):
        mock_highs.return_value = "highs-solver"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_pulp_solver("highs"), "highs-solver")
        mock_highs.assert_called_once_with(msg=0)


class TestEndToEnd(unittest.TestCase):
    """Solve a tiny ILP through the helpers with the installed PuLP."""

    def test_small_ilp(self):
        if not pulp.listSolvers(onlyAvailable=True):
            self.skipTest("no PuLP solver available")

        prob = pulp.LpProblem("compatibility_test", pulp.LpMinimize)
        x = _new_var_dict(prob, "x", [(i, j) for i in range(2) for j in range(2)],
                          cat="Binary")
        t = _new_var(prob, "t", cat="Integer")
        prob += t
        prob += pulp.lpSum(x.values()) >= 3
        prob += t >= pulp.lpSum(x.values())

        code, status = _solve(prob, select_pulp_solver())

        self.assertEqual((code, status), (1, "Optimal"))
        self.assertEqual(round(pulp.value(t)), 3)
        self.assertEqual(sum(round(pulp.value(v)) for v in x.values()), 3)


if __name__ == "__main__":
    unittest.main()
