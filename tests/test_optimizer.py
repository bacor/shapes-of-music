import unittest
import numpy as np
from music21 import converter
from src.contour import Contour
from src.typology.optimize import iter_param_grid
from src.typology import HuronTypology
from src.typology import TypologyOptimizer


class TestOptimizer(unittest.TestCase):
    def test_param_grid(self):
        param_grid = dict(a=(0, 10, 2), b=(0, 90, 3))
        params = list(iter_param_grid(param_grid))
        target = [
            {"a": 0.0, "b": 0.0},
            {"a": 0.0, "b": 45.0},
            {"a": 0.0, "b": 90.0},
            {"a": 10.0, "b": 0.0},
            {"a": 10.0, "b": 45.0},
            {"a": 10.0, "b": 90.0},
        ]
        self.assertListEqual(params, target)

    def test_param_grid_direct(self):
        # numpy array
        param_grid = dict(a=np.array([0, 5]), b=(0, 90, 3))
        params = list(iter_param_grid(param_grid))
        target = [
            {"a": 0, "b": 0.0},
            {"a": 0, "b": 45.0},
            {"a": 0, "b": 90.0},
            {"a": 5, "b": 0.0},
            {"a": 5, "b": 45.0},
            {"a": 5, "b": 90.0},
        ]
        self.assertListEqual(params, target)

        # List
        param_grid = dict(a=[0, 5], b=(0, 90, 3))
        params = list(iter_param_grid(param_grid))
        self.assertListEqual(params, target)

    def test_grid_search(self):
        optimizer = TypologyOptimizer(HuronTypology)
        contours = [Contour([3, 2, 1]), Contour([1, 2, 3])]
        param_grid = dict(tolerance=[0, 0.5, 1])
        params, score = optimizer.grid_search(contours, param_grid)
        # Entropy (base 2) for two equally likely alternatives:
        self.assertEqual(score, 1)

    def test_optimize(self):
        optimizer = TypologyOptimizer(HuronTypology)
        contours = [Contour([3, 2, 1]), Contour([1, 2, 3]), Contour([2, 2, 2])]
        best, res = optimizer.optimize(contours, dict(tolerance=[0, 10]))

    def test_custom_scoring_fn(self):
        score_fn_1 = lambda typology, contours: 1
        score_fn_2 = lambda typology, contours: dict(score=1)
        contours = [Contour([3, 2, 1]), Contour([1, 2, 3]), Contour([2, 2, 2])]
        optimizer = TypologyOptimizer(HuronTypology, scoring_fn=score_fn_1)
        f = optimizer.loss_function(contours, ["tolerance"])
        self.assertEqual(f([0]), -1)

        optimizer2 = TypologyOptimizer(HuronTypology, scoring_fn=score_fn_2)
        g = optimizer2.loss_function(contours, ["tolerance"])
        self.assertEqual(g([0]), -1)
