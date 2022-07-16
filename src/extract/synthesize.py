import os
from typing import Tuple, Optional, Union
from collections import Counter
import scipy.stats
import numpy as np
import pandas as pd

from ..contour import Contour
from ..config import CONTOUR_DIR
from .extract_contours import store_shuffled_indices


def binom(mean: float, var: float) -> scipy.stats.binom:
    """Construct a binomial distribution from its mean and variance

    Parameters
    ----------
    mean : float
        The mean
    var : float
        Variance

    Returns
    -------
    scipy.stats.binom
        A binomial distribution
    """
    p = 1 - var / mean
    n = int(mean / p)
    return scipy.stats.binom(n=n, p=p)


def sample(
    distribution: Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete]
) -> Union[float, int]:
    """Sample a single point from a distribution

    Parameters
    ----------
    distribution : Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete]
        The distribution

    Returns
    -------
    Union[float, int]
        A sample from the distribution
    """
    return distribution.rvs(size=1)[0]


class ContourSynthesizer:
    contour_id_tmpl = "synth-{i:0>4}"

    initial_dist = None

    length_dist = None

    def estimate_length_distribution(self, unit_lengths: np.array):
        mu = unit_lengths.mean()
        return scipy.stats.poisson(mu)

    def fit(self, *args):
        raise NotImplemented

    def generate_initial(self):
        raise NotImplemented

    def step(self, i: int, contour: np.array) -> float:
        raise NotImplemented

    def generate(self):
        length = max(3, sample(self.length_distr))
        contour = np.zeros(length)
        contour[0] = self.generate_initial()
        for i in range(1, length):
            contour[i] = self.step(i, contour)
        return contour

    def generate_dataset(
        self,
        num_contours: int,
        num_samples: Optional[int] = 50,
        random_state: int = 12345,
    ) -> pd.DataFrame:
        """"""
        np.random.seed(random_state)
        contours = np.zeros((num_contours, num_samples))
        lengths = np.zeros((num_contours, 1))
        ids = []
        for i in range(num_contours):
            contour = Contour(self.generate())
            lengths[i] = len(contour)
            ids.append(self.contour_id_tmpl.format(i=i))
            contours[i, :] = contour.interpolate(num_samples).pitches

        # Collect in dataframe
        df = pd.DataFrame(contours, index=ids, columns=range(num_samples))
        df.index.name = "contour_id"
        df["song_id"] = None
        df["unit_num"] = None
        df["unit_length"] = lengths
        df["unit_duration"] = lengths
        df["tonic_krumhansl"] = None
        df["tonic_mode"] = None
        df["final"] = contours[:, -1]
        df["mode"] = None
        columns = [
            "song_id",
            "unit_num",
            "unit_length",
            "unit_duration",
            "tonic_krumhansl",
            "tonic_mode",
            "final",
            "mode",
        ] + [i for i in range(num_samples)]
        df = df.reindex(columns=columns)
        return df


class MarkovSynthesizer(ContourSynthesizer):
    contour_id_tmpl = "markov-{i:0>4}"

    def __init__(self, smoothing: Optional[int] = 1):
        self.smoothing = smoothing

    def count_bigrams(
        self, contours: np.array, count_repetitions: Optional[bool] = False
    ) -> Counter:
        counts = Counter()
        for contour in contours:
            for p1, p2 in zip(contour[:-1], contour[1:]):
                if count_repetitions or p1 != p2:
                    counts[(p1, p2)] += 1
        return counts

    def estimate_transition_matrix(
        self, contours: np.array
    ) -> Tuple[np.array, np.array]:
        domain = np.arange(contours.min(), contours.max())
        transitions = np.zeros((len(domain), len(domain)))
        bigrams = self.count_bigrams(contours)
        for i, p1 in enumerate(domain):
            for j, p2 in enumerate(domain):
                transitions[i, j] = bigrams[(p1, p2)] + self.smoothing
        transitions = transitions / transitions.sum(axis=1)[:, None]
        transitions = pd.DataFrame(transitions, columns=domain, index=domain)
        return domain, transitions

    def estimate_initial_distribution(self, contours: np.array) -> scipy.stats.binom:
        initials = contours[:, 0]
        mean = np.round(initials.mean())
        var = initials.var()
        return binom(mean, var)

    def fit(self, contours, unit_lengths):
        self.initial_distr = self.estimate_initial_distribution(contours)
        self.length_distr = self.estimate_length_distribution(unit_lengths)
        self.domain, self.transition_matrix = self.estimate_transition_matrix(contours)

    def generate_initial(self):
        initial = None
        while initial is None:
            candidate = sample(self.initial_distr)
            if candidate in self.domain:
                return candidate

    def step(self, i: int, contour: np.array) -> float:
        probs = self.transition_matrix.loc[contour[i - 1], :].values
        return np.random.choice(self.domain, size=1, p=probs)[0]


class BinomialWalkSynthesizer(ContourSynthesizer):
    contour_id_tmpl = "binom-{i:0>4}"

    def __init__(
        self, n: int = 10, p: float = 0.5, min_pitch: int = 60, max_pitch: int = 84
    ):
        self.n = n
        self.p = p
        self.domain = np.arange(min_pitch, max_pitch + 1)

    def fit(self, unit_lengths: np.array):
        self.length_distr = self.estimate_length_distribution(unit_lengths)

    def generate_initial(self):
        return np.random.choice(self.domain, size=1)

    def step(self, i: int, contour: np.array) -> np.array:
        step = np.random.binomial(self.n, self.p) - self.n * self.p
        step = min(max(-12, step), 12)
        if (contour[i - 1] + step > 84) or (contour[i - 1] + step < 60):
            return contour[i - 1] - step
        else:
            return contour[i - 1] + step


def main():
    import argparse
    from ..clusterability import Dataset

    parser = argparse.ArgumentParser(
        description="Generate a dataset of synthetic Markov contours"
    )
    parser.add_argument(
        "synthesizer",
        type=str,
        help="The dataset on which to fit the Markov contours",
    )
    args = parser.parse_args()

    dataset = Dataset("combined-phrase")
    contours = dataset.representation("pitch")

    # Fit synthesizer
    if args.synthesizer == "markov":
        synthesizer = MarkovSynthesizer()
        synthesizer.fit(contours, dataset.df["unit_length"])
    elif args.synthesizer == "binom":
        synthesizer = BinomialWalkSynthesizer()
        synthesizer.fit(dataset.df["unit_length"])

    # Generate contours dataset
    synth_contours = synthesizer.generate_dataset(num_contours=5000)

    # Save
    contours_fn = os.path.join(CONTOUR_DIR, f"{args.synthesizer}-contours.csv.gz")
    synth_contours.to_csv(contours_fn, compression="gzip")
    subset_fn = os.path.join(CONTOUR_DIR, f"{args.synthesizer}-contours-indices.json")
    store_shuffled_indices(synth_contours, subset_fn)


if __name__ == "__main__":
    main()
