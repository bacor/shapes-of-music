import numpy as np
from .boundary import BoundaryTypology


class AdamsTypology(BoundaryTypology):
    types = [
        "descending-both",
        "horizontal-both",
        "ascending-both",
        "descending-lowest-ascending",
        "horizontal-lowest-ascending",
        "ascending-lowest-ascending",
        "descending-highest-descending",
        "horizontal-highest-descending",
        "ascending-highest-descending",
        "descending-neither-ascending",
        "horizontal-neither-ascending",
        "ascending-neither-ascending",
        "descending-neither-descending",
        "horizontal-neither-descending",
        "ascending-neither-descending",
    ]

    def boundary(self, contour, as_list=False):
        boundary = super().boundary(contour, as_list=True)
        if contour.initial == contour.highest or contour.final == contour.highest:
            boundary.pop(1)
        if contour.initial == contour.lowest or contour.final == contour.lowest:
            boundary.pop(-2)

        if len(boundary) == 4:
            max_index = contour.pitches.argmax()
            min_index = contour.pitches.argmin()
            if max_index > min_index:
                boundary = [boundary[0], boundary[2], boundary[1], boundary[3]]

        if as_list:
            return boundary
        else:
            return "".join(str(b) for b in boundary)

    def opening(self, contour):
        boundary = self.boundary(contour, as_list=True)
        if boundary[0] > boundary[1] + self.tolerance:
            return "descending"
        elif boundary[0] < boundary[1] - self.tolerance:
            return "ascending"
        else:
            return "horizontal"

    def slope_num(self, contour):
        numbering = dict(descending=1, horizontal=2, ascending=3)
        return numbering[self.slope(contour)]

    def deviations(self, contour):
        deviations = 0
        for x in [contour.lowest, contour.highest]:
            x_is_I = np.abs(x - contour.initial) <= self.tolerance
            x_is_F = np.abs(x - contour.final) <= self.tolerance
            if not x_is_I and not x_is_F:
                deviations += 1
        return deviations

    def reciprocal(self, contour):
        boundary = self.boundary(contour, as_list=True)
        if len(boundary) == 2:
            return "none"
        if boundary[0] > boundary[1] + self.tolerance:
            return "descending"
        elif boundary[0] < boundary[1] - self.tolerance:
            return "ascending"

    def reciprocal_num(self, contour):
        numbering = dict(none=0, ascending=1, descending=2)
        return numbering[self.reciprocal(contour)]

    # TODO: now 24 classes!
    def classify(self, contour, feature_based=True, adams_names=False):
        if adams_names:
            S = self.slope_num(contour)
            D = self.deviations(contour)
            R = self.reciprocal_num(contour)
            return f"S{S} D{D} R{R}"
        elif feature_based:
            slope = self.slope(contour)
            scope = self.scope(contour)
            opening = self.opening(contour)
            if scope == "both":
                return f"{slope}-{scope}"
            else:
                return f"{slope}-{scope}-{opening}"
        else:
            return self.boundary(contour, as_list=False)
