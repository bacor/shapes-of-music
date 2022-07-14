import numpy as np
from .typology import Typology


class BoundaryTypology(Typology):
    types = [
        "descending-both",
        "descending-lowest",
        "descending-highest",
        "descending-neither",
        "horizontal-both",
        "horizontal-lowest",
        "horizontal-highest",
        "horizontal-neither",
        "ascending-both",
        "ascending-lowest",
        "ascending-highest",
        "ascending-neither",
    ]

    def __init__(self, tolerance=0):
        self.tolerance = tolerance

    def boundary(self, contour, as_list=False):
        pitches = [contour.initial, contour.highest, contour.lowest, contour.final]
        values = sorted(list(set(pitches)))
        boundary = [values.index(b) + 1 for b in pitches]
        if as_list:
            return boundary
        else:
            return "".join(str(b) for b in boundary)

    def slope(self, contour):
        if contour.initial > contour.final + self.tolerance:
            return "descending"
        elif contour.initial < contour.final - self.tolerance:
            return "ascending"
        else:
            return "horizontal"

    def scope(self, contour):
        I_is_H = np.abs(contour.initial - contour.highest) <= self.tolerance
        I_is_L = np.abs(contour.initial - contour.lowest) <= self.tolerance
        F_is_H = np.abs(contour.final - contour.highest) <= self.tolerance
        F_is_L = np.abs(contour.final - contour.lowest) <= self.tolerance

        if not I_is_H and not I_is_L and not F_is_H and not F_is_L:
            return "neither"
        elif (I_is_H or F_is_H) and not (I_is_L or F_is_L):
            return "highest"
        elif (I_is_L or F_is_L) and not (I_is_H or F_is_H):
            return "lowest"
        else:
            return "both"

    def classify(self, contour, feature_based=True):
        if feature_based:
            slope = self.slope(contour)
            scope = self.scope(contour)
            return f"{slope}-{scope}"
        else:
            return self.boundary(contour, as_list=False)
