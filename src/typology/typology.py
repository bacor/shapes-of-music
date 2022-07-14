from collections import Counter
import numpy as np


class Typology:
    """Abstract Typology base class"""

    types = None

    params = dict()

    def __len__(self):
        return len(self.types)

    def type_dist(self, contours):
        """Estimate the type distribution from a collection of contours.

        Parameters
        ----------
        contours : iterable
            An iterable of Contour objects

        Returns
        -------
        np.array
            An array of type frequencies of the same length as the number of
            types in the typology.
        """
        classes = [self.classify(c) for c in contours]
        counts = Counter(classes)
        freqs = np.asarray([counts[t] for t in self.types]) / len(contours)
        return freqs

    def classify(self, contour) -> str:
        """Classify a contour to one of the types used by the typology.

        Parameters
        ----------
        contour : Contour
            The contour

        Returns
        -------
        type : str
            The label of the type or class
        """
        raise NotImplemented
