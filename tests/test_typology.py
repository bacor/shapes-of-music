import unittest
from src.typology import huron_contour_type

class TestHuronTypology(unittest.TestCase):

    def test_types(self):
        h = huron_contour_type
        self.assertEqual(h([0, 0, 0]), 'horizontal')
        self.assertEqual(h([0, 1, 2]), 'ascending')
        self.assertEqual(h([0,-1,-2]), 'descending')
        self.assertEqual(h([0, 1, 0]), 'convex')
        self.assertEqual(h([0,-1, 0]), 'concave')
        self.assertEqual(h([0, 0, 1]), 'horizontal-ascending')
        self.assertEqual(h([0, 0,-1]), 'horizontal-descending')
        self.assertEqual(h([0, 1, 1]), 'ascending-horizontal')
        self.assertEqual(h([0,-1,-1]), 'descending-horizontal')

    def test_tolerance(self):
        h = huron_contour_type
        self.assertEqual(h([0, 1, -.5], tol=2), 'horizontal')
        self.assertEqual(h([0, 0.1, 0], tol=.05), 'convex')
    
if __name__ == '__main__':
    unittest.main()    