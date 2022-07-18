import os
import logging
import json
import numpy as np
import pandas as pd
import h5py
from itertools import product
from typing import List, Optional, Dict

from scipy.spatial.distance import pdist, squareform
from tslearn.metrics import cdist_dtw
from scipy.stats import gaussian_kde

from ..config import CONTOUR_DIR, SERIALIZED_DIR
from .representations import *
from .representations import contour_array

CONDITIONS = {
    "pitch": ["eucl", "dtw"],
    "pitch_centered": ["eucl", "dtw"],
    "pitch_normalized": ["eucl", "dtw"],
    "pitch_tonicized": ["eucl", "dtw"],
    "pitch_finalized": ["eucl", "dtw"],
    "cosine": ["eucl"],
    "interval": ["eucl", "dtw"],
    "smooth_derivative": ["eucl", "dtw"],
}
PRECOMPUTED_CONDITIONS = [c for c in CONDITIONS.keys() if c != "cosine"]
PRECOMPUTED_METRICS = ["dtw"]
PRECOMPUTED_LENGTHS = [None, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
PRECOMPUTED_LIMIT = 500


def save(obj: np.array, name: str, file: h5py.File, refresh: Optional[bool] = False):
    """Save an numpy array to a hdf5 file

    Parameters
    ----------
    obj : np.array
        The array
    name : str
        Name where to save it
    file : h5py.File
        The open file
    refresh : Optional[bool], optional
        Whether to refresh it, by default False
    """
    if refresh and name in file.keys():
        del file[name]
    file.create_dataset(name, obj.shape, obj.dtype, obj)


class Dataset(object):
    def __init__(self, dataset: str, refresh: Optional[bool] = False):
        self.dataset = dataset
        self._df = None

        # Load subsets
        indices_fn = os.path.join(CONTOUR_DIR, f"{dataset}-contours-indices.json")
        with open(indices_fn, "r") as handle:
            self.indices = json.load(handle)

        # Load contours
        self.fn = os.path.join(SERIALIZED_DIR, f"{dataset}.h5")
        with h5py.File(self.fn, "a") as file:
            if "contours" not in file.keys() or refresh:
                self.log("Extracting contours and metadata columns.")
                contours = contour_array(self.df)
                assert np.isinf(contours).any() == False
                assert np.isnan(contours).any() == False
                save(contours, "contours", file, refresh=refresh)

                for col in ["tonic_krumhansl", "tonic_mode", "final", "unit_length"]:
                    column = self.df[col].values
                    save(column, f"meta/{col}", file, refresh=refresh)

            self.num_contours = file["contours"].shape[0]
            self.num_samples = file["contours"].shape[1]

    def __repr__(self) -> str:
        return f"<Dataset {self.dataset}>"

    def log(self, message):
        logging.info(f"[{self.dataset}] {message}")

    @property
    def df(self):
        if self._df is None:
            df_fn = os.path.join(CONTOUR_DIR, f"{self.dataset}-contours.csv.gz")
            self._df = pd.read_csv(df_fn, index_col=0)
        return self._df

    ### Subsets

    def subset_index(
        self,
        length: Optional[int] = None,
        unique: Optional[bool] = False,
        limit: Optional[int] = None,
    ) -> List[int]:
        """Return the index for subset of the dataset. You can specify whether
        to return all contours or only the unique ones, and filter contours of
        a particular unit length. You can also limit the number of contours
        returned. Note that the returned index is always shuffled, using the
        order stored in the `[dataset-name]-indices.json` file.

        Parameters
        ----------
        length : int, optional
            Return only units of a particular length, by default None
        unique : bool, optional
            Only return unique contours, by default False
        limit : int, optional
            Return at most limit units, by default None

        Returns
        -------
        list
            A list of contour indices
        """
        indices = self.indices["all"] if unique == False else self.indices["unique"]
        if length is not None:
            with h5py.File(self.fn, "r") as file:
                (matches,) = np.where(file["meta/unit_length"][:] == length)
            indices = [i for i in indices if i in matches]
        return np.sort(indices[:limit])

    def subset_name(
        self,
        length: Optional[int] = None,
        unique: Optional[bool] = False,
        limit: Optional[int] = None,
    ) -> str:
        """Return a name for the subset that is used for storing data in the
        hd5 store.

        Parameters
        ----------
        length : Optional[int], optional
            The length of the contour, by default None
        unique : Optional[bool], optional
            Whether to return only unique contours, by default False
        limit : Optional[int], optional
            The maximum number of contours returned, by default None

        Returns
        -------
        str
            A name for the subset
        """
        subset = "unique" if unique else "all"
        length = "all" if length is None else length
        limit = "none" if limit is None else limit
        return f"subset-{subset}/length-{length}/limit-{limit}"

    def subset_size(self, **subset_kwargs) -> int:
        """Return the size of a subset

        Returns
        -------
        int
            the length of the subset
        """
        subset_index = self.subset_index(**subset_kwargs)
        return len(subset_index)

    def subset_column(self, column: str, **subset_kwargs):
        index = self.subset_index(**subset_kwargs)
        column = self.df[column].values
        return column[index]

    def representation(self, name: str, **subset_kwargs) -> np.array:
        """Return an array of contours in a certain representation. You can pass
        keyword arguments to select a particular subset of contours.

        Parameters
        ----------
        name : str
            The name of the representation, e.g. 'pitch'

        Returns
        -------
        np.array
            A numpy array of shape (num_contours, num_samples)
        """
        if name not in CONDITIONS.keys():
            raise Exception(f'Unknown representation "{name}"')

        index = self.subset_index(**subset_kwargs)
        if len(index) == 0:
            return np.zeros((0, self.num_samples))
        else:
            with h5py.File(self.fn, "r+") as file:
                repr_fn = globals()[f"repr_{name}"]
                contours = repr_fn(
                    file["contours"][index, :],
                    final=file["meta/final"][index],
                    tonic_krumhansl=file["meta/tonic_krumhansl"][index],
                    tonic_mode=file["meta/tonic_mode"][index],
                )[:]
                if np.isinf(contours).any():
                    logging.warn("Some contours contain np.inf")
                if np.isinf(contours).any():
                    logging.warn("Some contours contain np.nan")
            return contours

    def similarities(
        self,
        representation: str,
        metric: str,
        dtw_kws: Optional[Dict] = {},
        refresh: Optional[bool] = False,
        **subset_kwargs,
    ) -> np.array:
        """_summary_

        Parameters
        ----------
        representation : str
            The representation to use
        metric : str
            The metric (e.g. eucl or dtw)
        dtw_kws : Dict, optional
            Optional keyword arguments for the dtw distance, by default {}

        Returns
        -------
        np.array
            An array of similarity scores in squareform.
        """
        if representation not in CONDITIONS:
            raise ValueError(f'Unknown representation "{representation}"')
        if metric not in CONDITIONS[representation]:
            raise ValueError(
                f'Unsupported metric "{metric}" for representation "{representation}"'
            )
        if metric == "eucl":
            return self.eucl_similarities(representation, **subset_kwargs)
        elif metric == "dtw":
            return self.dtw_similarities(
                representation, refresh=refresh, **dtw_kws, **subset_kwargs
            )

    def eucl_similarities(self, representation: str, **subset_kwargs) -> np.array:
        """Compute Euclidean similarities between contours. These are not stored
        as computation is very efficient.

        Parameters
        ----------
        representation : str
            The representation

        Returns
        -------
        np.array
            An array with similarities
        """
        contours = self.representation(representation, **subset_kwargs)
        if contours.shape[0] > 0:
            similarities = pdist(contours, metric="euclidean")[:]
        else:
            similarities = np.zeros((0,))
        return similarities

    def dtw_similarities(
        self,
        representation: str,
        refresh: Optional[bool] = False,
        serialize: Optional[bool] = True,
        global_constraint: Optional[str] = "sakoe_chiba",
        sakoe_chiba_radius: Optional[float] = 20,
        **subset_kwargs,
    ):
        """"""
        subset_name = self.subset_name(**subset_kwargs)
        name = f"dtw-similarity/{representation}/{subset_name}"
        with h5py.File(self.fn, "r+") as file:
            if name in file.keys() and not refresh:
                similarities = file[name][:]
            else:
                self.log(f"Computing DTW similarities:")
                self.log(f"> name = {name}")
                contours = self.representation(representation, **subset_kwargs)
                if contours.shape[0] > 0:
                    similarities = cdist_dtw(
                        contours,
                        global_constraint=global_constraint,
                        sakoe_chiba_radius=sakoe_chiba_radius,
                    )
                    similarities = squareform(similarities)
                else:
                    similarities = np.zeros((0,))

                if serialize:
                    save(similarities, name, file, refresh=refresh)

        return similarities

    def similarity_kde(
        self,
        representation: str,
        metric: str,
        num_points: Optional[int] = 2000,
        refresh: Optional[bool] = False,
        serialize: Optional[bool] = True,
        dtw_kws: Optional[Dict] = {},
        **subset_kwargs,
    ) -> np.array:
        """"""
        subset_name = self.subset_name(**subset_kwargs)
        name = f"kde/{representation}/{metric}/{subset_name}"
        error_occured = False
        with h5py.File(self.fn, "r+") as file:
            if name in file.keys() and not refresh:
                distribution = file[name][:]

            else:
                sim = self.similarities(
                    representation, metric, dtw_kws=dtw_kws, **subset_kwargs
                )
                xs = np.array([])
                ys = np.array([])
                if len(sim) > 1:
                    try:
                        kde = gaussian_kde(sim)
                        margin = (sim.max() - sim.min()) * 0.05
                        xs = np.linspace(
                            sim.min() - margin, sim.max() + margin, num_points
                        )
                        ys = kde(xs)
                    except Exception as e:
                        S = squareform(sim)
                        contours = self.representation(representation, **subset_kwargs)

                        logging.error(f"An error occured: {e}")
                        logging.error(f"Similarity matrix: {sim}")
                        error_occured = True

                distribution = np.c_[xs, ys]
                if serialize and not error_occured:
                    save(distribution, name, file, refresh=refresh)

        if error_occured:
            return False
        else:
            return distribution

    # def similarity_hist(self, *args, **kwargs):
    #     sim = self.similarities(*args, **kwargs)
    #     hist, bins = np.histogram(sim, bins="auto", density=True)
    #     return hist, bins

    # def clean_similarity_kde(self):
    #     with h5py.File(self.fn, "a") as file:
    #         if "kde" in file.keys():
    #             del file["kde"]

    # def hartigans_dip(self, *args, **kwargs, refresh=False):
    #     sim = self.similarities(*args, **kwargs)
    #     with h5py.File(self.fn, 'a') as file:
    #         subset_name = self.subset_name(**subset_kwargs)
    #         name = f'kde/{representation}/{metric}/{subset_name}'

    def precompute_all(self, refresh: Optional[bool] = False):
        for repres, metric, length, unique in product(
            PRECOMPUTED_CONDITIONS,
            PRECOMPUTED_METRICS,
            PRECOMPUTED_LENGTHS,
            [True, False],
        ):
            # Hacky, but skip these
            if (
                self.dataset == "markov" or self.dataset == "binom"
            ) and repres == "pitch_tonicized":
                continue

            settings = dict(
                representation=repres,
                metric=metric,
                unique=unique,
                length=length,
                limit=PRECOMPUTED_LIMIT,
                refresh=refresh,
            )
            self.similarities(**settings)
            self.similarity_kde(**settings)


if __name__ == "__main__":
    dataset = Dataset("markov", refresh=True)
    contours = dataset.representation("pitch", limit=100)
    pass
    # dataset = Dataset("liber-antiphons-phrase", refresh=True)
    # dataset.precompute_all()
