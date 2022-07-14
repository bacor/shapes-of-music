import numpy as np
from .typology import Typology
from ..contour import Contour


class HuronTypology(Typology):
    types = [
        "descending",
        "ascending",
        "convex",
        "concave",
        "horizontal",
        "horizontal-descending",
        "horizontal-ascending",
        "descending-horizontal",
        "ascending-horizontal",
    ]

    params = {"tolerance": [0, np.inf], "bound_1": [0, 1], "bound_2": [0, 1]}

    def __init__(self, tolerance, bound_1=1 / 3, bound_2=2 / 3):
        """Hurons typology

        Parameters
        ----------
        tolerance : float
            The tolerance parameter
        bound_1 : float, optional
            The boundary between the beginning and middle parts, by default 1/3
        bound_2 : float, optional
            The boundary between the middle and final part, by default 2/3
        """
        self.tolerance = tolerance
        self.bound_1 = bound_1
        self.bound_2 = bound_2

    def classify_parts(self, begin, middle, end):
        """Get the type of a contour based on its three parts: the beginning,
        middle and ending. These parts can be averaged pitches, or single
        pitches. This method is used by the classify method.

        Parameters
        ----------
        begin : float
            (Average) pitch of the first part
        middle : float
            (Average) pitch of the middle part
        end : float
            (Average) pitch of the final part

        Returns
        -------
        str
            Label of the type
        """
        beginAboveMiddle = begin - middle > self.tolerance
        beginEqualMiddle = np.abs(begin - middle) <= self.tolerance
        beginBelowMiddle = middle - begin > self.tolerance
        middleAboveEnd = middle - end > self.tolerance
        middleEqualEnd = np.abs(middle - end) <= self.tolerance
        middleBelowEnd = end - middle > self.tolerance

        if beginBelowMiddle and middleBelowEnd:
            contour_type = "ascending"
        elif beginAboveMiddle and middleAboveEnd:
            contour_type = "descending"
        elif beginAboveMiddle and middleBelowEnd:
            contour_type = "concave"
        elif beginBelowMiddle and middleAboveEnd:
            contour_type = "convex"
        elif beginEqualMiddle and middleBelowEnd:
            contour_type = "horizontal-ascending"
        elif beginEqualMiddle and middleAboveEnd:
            contour_type = "horizontal-descending"
        elif beginBelowMiddle and middleEqualEnd:
            contour_type = "ascending-horizontal"
        elif beginAboveMiddle and middleEqualEnd:
            contour_type = "descending-horizontal"
        elif beginEqualMiddle and middleEqualEnd:
            contour_type = "horizontal"
        else:
            raise Exception("This should not be possible")

        return contour_type

    def classify(self, contour: Contour):
        pos = contour.positions
        begin = contour.pitches[pos < self.bound_1].mean()
        is_middle = (self.bound_1 <= pos) & (pos < self.bound_2)
        middle = contour.pitches[is_middle].mean()
        end = contour.pitches[self.bound_2 <= pos].mean()
        return self.classify_parts(begin, middle, end)


class HuronEndpointTypology(HuronTypology):
    """The extreme version of Huron's typology, which only compares the initial
    and final note with the average pitch of everything in beweteen."""

    def classify(self, contour):
        begin = contour.pitches[0]
        end = contour.pitches[-1]
        middle = contour.pitches[1:-1].mean()
        return self.classify_parts(begin, middle, end)
