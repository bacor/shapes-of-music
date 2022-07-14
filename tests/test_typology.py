import unittest
import numpy as np
from src.contour import Contour
from src.typology import HuronTypology
from src.typology import HuronEndpointTypology
from src.typology import BoundaryTypology
from src.typology import AdamsTypology


class TestHuronTypology(unittest.TestCase):
    def test_temporal_slicing(self):
        """The contour should be divided in three parts of equal *duration*,
        not of equal length (equal number of points). This test makes sure
        that is the case"""
        # Three parts of equal length:
        # Pitches:   1 3 | 2 1 | 5 5
        # Averages:   2  | 1.5 | 5
        # Type:      concave

        # If parts of equal duration:
        # Pitches:   1 3 2 | 1 5 | 5
        # Averages:    2   |  3  | 5
        # Type:      ascending
        contour = Contour([1, 3, 2, 1, 5, 5], [1, 2, 3, 4, 5, 10])
        typology = HuronTypology(tolerance=0)
        self.assertEqual(typology.classify(contour), "ascending")

    def test_types(self):
        """Test all contour types using contours with three pitches"""
        typology = HuronTypology(tolerance=0.2)

        def h(pitches):
            contour = Contour(pitches)
            return typology.classify(contour)

        self.assertEqual(h([0, 0, 0]), "horizontal")
        self.assertEqual(h([0, 1, 2]), "ascending")
        self.assertEqual(h([0, -1, -2]), "descending")
        self.assertEqual(h([0, 1, 0]), "convex")
        self.assertEqual(h([0, -1, 0]), "concave")
        self.assertEqual(h([0, 0, 1]), "horizontal-ascending")
        self.assertEqual(h([0, 0, -1]), "horizontal-descending")
        self.assertEqual(h([0, 1, 1]), "ascending-horizontal")
        self.assertEqual(h([0, -1, -1]), "descending-horizontal")

    def test_tolerance(self):
        def h(pitches, tol):
            typology = HuronTypology(tolerance=tol)
            contour = Contour(pitches)
            return typology.classify(contour)

        self.assertEqual(h([0, 1, -0.5], tol=2), "horizontal")
        self.assertEqual(h([0, 0.1, 0], tol=0.05), "convex")

    def test_endpoint_typology(self):
        """Test whether Hurons 'extreme' typology is indeed different from the
        normal Huron typology by checking a contour which has different
        contour types under both typologies"""
        # 1 3 1 1 3 1
        # Normal: (2, 1, 2) --> concave
        # Crude:  (1, 2, 1) --> convex
        contour = Contour([1, 3, 1, 1, 3, 1])
        typology = HuronTypology(tolerance=0.2)
        self.assertEqual(typology.classify(contour), "concave")
        crude_typology = HuronEndpointTypology(tolerance=0.2)
        self.assertEqual(crude_typology.classify(contour), "convex")


class TestBoundTypology(unittest.TestCase):
    def test_all_types(self):
        typology = BoundaryTypology(tolerance=0)

        # scope = both

        c2211 = Contour([2, 2, 1, 1])
        self.assertEqual(typology.slope(c2211), "descending")
        self.assertEqual(typology.scope(c2211), "both")
        self.assertEqual(typology.classify(c2211), "descending-both")
        self.assertEqual(typology.classify(c2211, feature_based=False), "2211")

        c1111 = Contour([1, 1, 1, 1])
        self.assertEqual(typology.slope(c1111), "horizontal")
        self.assertEqual(typology.scope(c1111), "both")
        self.assertEqual(typology.classify(c1111), "horizontal-both")
        self.assertEqual(typology.classify(c1111, feature_based=False), "1111")

        c1212 = Contour([1, 2, 1, 2])
        self.assertEqual(typology.slope(c1212), "ascending")
        self.assertEqual(typology.scope(c1212), "both")
        self.assertEqual(typology.classify(c1212), "ascending-both")
        self.assertEqual(typology.classify(c1212, feature_based=False), "1212")

        # scope = lowest

        c2311 = Contour([2, 3, 1, 1])
        self.assertEqual(typology.slope(c2311), "descending")
        self.assertEqual(typology.scope(c2311), "lowest")
        self.assertEqual(typology.classify(c2311), "descending-lowest")
        self.assertEqual(typology.classify(c2311, feature_based=False), "2311")

        c1211 = Contour([1, 2, 1, 1])
        self.assertEqual(typology.slope(c1211), "horizontal")
        self.assertEqual(typology.scope(c1211), "lowest")
        self.assertEqual(typology.classify(c1211), "horizontal-lowest")
        self.assertEqual(typology.classify(c1211, feature_based=False), "1211")

        c1312 = Contour([1, 3, 1, 2])
        self.assertEqual(typology.slope(c1312), "ascending")
        self.assertEqual(typology.scope(c1312), "lowest")
        self.assertEqual(typology.classify(c1312), "ascending-lowest")
        self.assertEqual(typology.classify(c1312, feature_based=False), "1312")

        # scope = highest

        c3312 = Contour([3, 3, 1, 2])
        self.assertEqual(typology.slope(c3312), "descending")
        self.assertEqual(typology.scope(c3312), "highest")
        self.assertEqual(typology.classify(c3312), "descending-highest")
        self.assertEqual(typology.classify(c3312, feature_based=False), "3312")

        c2212 = Contour([2, 2, 1, 2])
        self.assertEqual(typology.slope(c2212), "horizontal")
        self.assertEqual(typology.scope(c2212), "highest")
        self.assertEqual(typology.classify(c2212), "horizontal-highest")
        self.assertEqual(typology.classify(c2212, feature_based=False), "2212")

        c2313 = Contour([2, 3, 1, 3])
        self.assertEqual(typology.slope(c2313), "ascending")
        self.assertEqual(typology.scope(c2313), "highest")
        self.assertEqual(typology.classify(c2313), "ascending-highest")
        self.assertEqual(typology.classify(c2313, feature_based=False), "2313")

        # scope = neither

        c3412 = Contour([3, 4, 1, 2])
        self.assertEqual(typology.slope(c3412), "descending")
        self.assertEqual(typology.scope(c3412), "neither")
        self.assertEqual(typology.classify(c3412), "descending-neither")
        self.assertEqual(typology.classify(c3412, feature_based=False), "3412")

        c2312 = Contour([2, 3, 1, 2])
        self.assertEqual(typology.slope(c2312), "horizontal")
        self.assertEqual(typology.scope(c2312), "neither")
        self.assertEqual(typology.classify(c2312), "horizontal-neither")
        self.assertEqual(typology.classify(c2312, feature_based=False), "2312")

        c2413 = Contour([2, 4, 1, 3])
        self.assertEqual(typology.slope(c2413), "ascending")
        self.assertEqual(typology.scope(c2413), "neither")
        self.assertEqual(typology.classify(c2413), "ascending-neither")
        self.assertEqual(typology.classify(c2413, feature_based=False), "2413")

    def test_boundary(self):
        typology = BoundaryTypology()
        c = Contour([67, 67, 64, 65, 64, 62, 60])
        self.assertEqual(typology.boundary(c), "2211")
        self.assertListEqual(typology.boundary(c, as_list=True), [2, 2, 1, 1])


class TestAdamsTypology(unittest.TestCase):
    def test_types(self):
        typology = AdamsTypology()

        c21 = Contour([2, 2, 1, 1])
        self.assertEqual(typology.boundary(c21), "21")
        self.assertEqual(typology.opening(c21), "descending")
        self.assertEqual(typology.classify(c21), "descending-both")
        self.assertEqual(typology.classify(c21, adams_names=True), "S1 D0 R0")

        c11 = Contour([1, 1, 1, 1])
        self.assertEqual(typology.boundary(c11), "11")
        self.assertEqual(typology.opening(c11), "horizontal")
        self.assertEqual(typology.classify(c11), "horizontal-both")
        self.assertEqual(typology.classify(c11, adams_names=True), "S2 D0 R0")

        c12 = Contour([1, 1, 2, 2])
        self.assertEqual(typology.boundary(c12), "12")
        self.assertEqual(typology.opening(c12), "ascending")
        self.assertEqual(typology.classify(c12), "ascending-both")
        self.assertEqual(typology.classify(c12, adams_names=True), "S3 D0 R0")

        c231 = Contour([2, 3, 1, 1])
        self.assertEqual(typology.boundary(c231), "231")
        self.assertEqual(typology.opening(c231), "ascending")
        self.assertEqual(typology.classify(c231), "descending-lowest-ascending")
        self.assertEqual(typology.classify(c231, adams_names=True), "S1 D1 R1")

        # ...

        c3412 = Contour([3, 4, 1, 2])
        self.assertEqual(typology.boundary(c3412), "3412")
        self.assertEqual(typology.opening(c3412), "ascending")
        self.assertEqual(typology.classify(c3412), "descending-neither-ascending")
        self.assertEqual(typology.classify(c3412, adams_names=True), "S1 D2 R1")

        c2312 = Contour([2, 3, 1, 2])
        self.assertEqual(typology.boundary(c2312), "2312")
        self.assertEqual(typology.opening(c2312), "ascending")
        self.assertEqual(typology.classify(c2312), "horizontal-neither-ascending")
        self.assertEqual(typology.classify(c2312, adams_names=True), "S2 D2 R1")

        c3142 = Contour([3, 1, 4, 2])
        self.assertEqual(typology.boundary(c3142), "3142")
        self.assertEqual(typology.opening(c3142), "descending")
        self.assertEqual(typology.classify(c3142), "descending-neither-descending")
        self.assertEqual(typology.classify(c3142, adams_names=True), "S1 D2 R2")

        c2132 = Contour([2, 1, 3, 2])
        self.assertEqual(typology.boundary(c2132), "2132")
        self.assertEqual(typology.opening(c2132), "descending")
        self.assertEqual(typology.classify(c2132), "horizontal-neither-descending")
        self.assertEqual(typology.classify(c2132, adams_names=True), "S2 D2 R2")

    def test_slope(self):
        typology = AdamsTypology()
        contour = Contour([1, 2, 3])
        self.assertEqual(typology.slope(contour), "ascending")
        self.assertEqual(typology.slope_num(contour), 3)
        contour = Contour([3, 2, 1])
        self.assertEqual(typology.slope(contour), "descending")
        self.assertEqual(typology.slope_num(contour), 1)
        contour = Contour([1, 2, 1])
        self.assertEqual(typology.slope(contour), "horizontal")
        self.assertEqual(typology.slope_num(contour), 2)

        typology = AdamsTypology(tolerance=2)
        contour = Contour([1, 2, 3])
        self.assertEqual(typology.slope(contour), "horizontal")

    def test_deviations(self):
        typology = AdamsTypology()
        contour = Contour([1, 2, 2, 3])
        self.assertEqual(typology.deviations(contour), 0)

        contour = Contour([1, 3, 2, 3])
        self.assertEqual(typology.deviations(contour), 0)

        contour = Contour([1, 4, 2, 3])
        self.assertEqual(typology.deviations(contour), 1)
        contour = Contour([1, 2, 0, 3])
        self.assertEqual(typology.deviations(contour), 1)

        contour = Contour([1, 4, 0, 3])
        self.assertEqual(typology.deviations(contour), 2)

    # def test_scope_both(self):
    #     typology = AdamsTypology()
    #     c1 = Contour([1, 2, 2, 3])
    #     c2 = Contour([1, 1, 2, 3])
    #     print(c1)


if __name__ == "__main__":
    unittest.main()
