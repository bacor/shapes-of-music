import os
import logging
from functools import wraps
from typing import Callable, List, Optional, Dict, Union, Tuple

from hashlib import md5
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from tslearn.metrics import cdist_dtw
from unidip.dip import dip_fn, diptst
from tableone.modality import (
    cum_distr,
    dip_and_closest_unimodal_from_cdf,
    dip_pval_tabinterpol,
)

import matplotlib.pyplot as plt
import umap
from scipy.fft import idct

from src import representations

from .visualize import (
    average_inv_contours,
    grid_around_points,
    show_umap_sideplot,
    show_umap_plot,
)

from .config import (
    FIGURES_DIR,
    ALL_METRICS,
    METRIC_LIMITS,
    MIN_CONTOUR_COUNT,
    SIM_DISTR_SAMPLE_SIZE,
    validate_condition,
)
from .dataset import Dataset


class InvalidConditionException(Exception):
    """Raised when the condition is invalid"""

    ...


class TooFewContoursException(Exception):
    """Raised when too few contours are found for a condition"""

    ...


def serialize(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        self,
        *args,
        refresh_serialized: Optional[bool] = False,
        serialize: Optional[bool] = True,
        **kwargs,
    ):
        name = func.__name__
        path = "/".join(self.path(what=name))

        # By default turn on serialization on a class level
        if not hasattr(self, "_serialization_enabled"):
            self._serialization_enabled = True

        if not self._serialization_enabled:
            return func(self, *args, **kwargs)
        elif self.dataset.exists(path) and not refresh_serialized:
            output = self.dataset.load(path)
        else:
            output = func(self, *args, **kwargs)
            if serialize:
                # If the output is not a dictionary, turn it into one
                if not isinstance(output, dict):
                    output = dict(_default=output)
                self.dataset.store(path, output, refresh=refresh_serialized)

        # If there's only one value in the dictionary, return that
        if isinstance(output, dict) and "_default" in output:
            output = output["_default"]
        return output

    return wrapper


def memoize(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        self,
        *args,
        refresh_memoized: Optional[bool] = False,
        memoize: Optional[bool] = True,
        **kwargs,
    ):
        name = func.__name__

        # By default turn on memoization on a class level
        if not hasattr(self, "_memoization_enabled"):
            self._memoization_enabled = True
        if not hasattr(self, "_memoized"):
            self._memoized = {}

        if not self._memoization_enabled:
            return func(self, *args, **kwargs)
        elif name in self._memoized and not refresh_memoized:
            return self._memoized[name]
        else:
            output = func(self, *args, **kwargs)
            if memoize:
                self._memoized[name] = output
            return output

    return wrapper


def catch_exceptions(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # By default turn on exception catching on a class level
        if not hasattr(self, "_exception_catching_enabled"):
            self._exception_catching_enabled = True

        if not self._exception_catching_enabled:
            return func(self, *args, *kwargs)
        else:
            try:
                return func(self, *args, *kwargs)
            except Exception as e:
                logging.error(f'An exception occured when executing "{func.__name__}":')
                logging.error(f"> {e.__class__.__name__}: {e}")
                return

    return wrapper


def create_file(output_dir: str, ext: str) -> Callable:
    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, refresh_file: Optional[bool] = False, **kwargs):
            name = func.__name__
            parts = self.path(what=name)
            basedir = os.path.join(output_dir, *parts[:-1])
            path = os.path.join(basedir, f"{parts[-1]}.{ext}")

            if os.path.exists(path) and not refresh_file:
                return False
            else:
                if not os.path.exists(basedir):
                    os.makedirs(basedir)
                return func(self, *args, path=path, **kwargs)

        return wrapper

    return decorate


def log_start_end(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        name = func.__name__
        self.log(f"Start running {name}")
        output = func(self, *args, **kwargs)
        self.log(f"> Done running {name}")
        return output

    return wrapper


####


class Condition(object):
    def __init__(
        self,
        dataset: str,
        representation: str,
        metric: Optional[str] = "eucl",
        limit: Optional[int] = None,
        length: Optional[int] = None,
        unique: Optional[int] = False,
        dimensionality: Optional[int] = 50,
        dtw_kws: Optional[Dict] = {},
        umap_embed_kws: Optional[Dict] = {},
        log: Optional[bool] = False,
    ):
        if limit is None:
            limit = METRIC_LIMITS[metric]

        self.dataset_id = dataset
        self.dataset = Dataset(self.dataset_id, log=log)
        self.representation = representation
        self.metric = metric
        self.length = length
        self.unique = unique
        self.limit = limit
        self.dimensionality = dimensionality

        if log:
            self.log("*" * 80)
            self.log(f"Initializing {repr(self)[1:-1]}")

        if metric not in ALL_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if not validate_condition(self.as_dict()):
            raise InvalidConditionException(
                "The given combination of options is not allowed. "
                "See the variable `INVALID_COMBINATIONS` in `config.py` for "
                "all invalid combinations."
            )
        if len(self) < MIN_CONTOUR_COUNT:
            raise TooFewContoursException(
                "Not enough contours were found for this condition. Only "
                f"{len(self)} contours were found, but at least "
                f"{MIN_CONTOUR_COUNT} are required."
            )

        # Store DTW options
        self.dtw_kws = dict(
            global_constraint="sakoe_chiba", sakoe_chiba_radius=20, n_jobs=4
        )
        self.dtw_kws.update(dtw_kws)

        # Store UMAP embedding options
        self.umap_embed_kws = dict(
            n_neighbors=30,
            min_dist=0.0,
            n_components=10,
            random_state=42,
        )
        self.umap_embed_kws.update(umap_embed_kws)

    def __repr__(self) -> str:
        """A string representation of the condition"""
        props = ""
        if self.unique is not False:
            props += f"unique "
        if self.length is not None:
            props += f"len={self.length} "
        if self.dimensionality != 50:
            props += f"dim={self.dimensionality} "
        return (
            f"<Condition {self.dataset_id} repr={self.representation} metric={self.metric} "
            f"{props}limit={self.limit}>"
        )

    @memoize
    def __len__(self) -> int:
        return self.dataset.subset_size(
            length=self.length,
            unique=self.unique,
            limit=self.limit,
        )

    def as_dict(self) -> Dict:
        """Return a dictionary representing the condition"""
        return dict(
            dataset=self.dataset_id,
            representation=self.representation,
            metric=self.metric,
            length=self.length,
            unique=self.unique,
            limit=self.limit,
            dimensionality=self.dimensionality,
        )

    def path(self, what: Optional[str] = None) -> List[str]:
        """Return the parts of a path that can be used to store data for this
        condition, either in a HDF5 store, or in a file, etc.

        Parameters
        ----------
        what : Optional[str], optional
            An optional string describing what is stored, by default None

        Returns
        -------
        List[str]
            The parts of a path for this condition
        """
        subset = "unique" if self.unique else "all"
        length = "all" if self.length is None else self.length
        limit = "none" if self.limit is None else self.limit
        dim = self.dimensionality
        parts = [] if what is None else [what]
        parts += [
            self.dataset_id,
            f"{self.representation}-{self.metric}",
            f"length_{length}",
            f"limit_{limit}-subset_{subset}-dim_{dim}",
        ]
        return parts

    @property
    def hash(self) -> str:
        """Compute a hash string from the path."""
        slug = "-".join(self.path())
        hash = md5(slug.encode("utf-8")).hexdigest()
        return hash

    def log(self, message: str):
        logging.info(f"[{self.hash[:5]}] {message}")

    def is_serialized(self, what: str) -> bool:
        path = "/".join(self.path(what=what))
        return self.dataset.exists(path)

    def file_path(self, what: str, output_dir: str, ext: str) -> bool:
        parts = self.path(what=what)
        path = os.path.join(output_dir, *parts[:-1], f"{parts[-1]}.{ext}")
        return path

    def figure_path(self, what: str) -> str:
        return self.file_path(what=what, output_dir=FIGURES_DIR, ext="pdf")

    @property
    def df(self):
        index = self.dataset.subset_index(
            length=self.length, unique=self.unique, limit=self.limit
        )
        return self.dataset.df.iloc[index, :]

    @memoize
    def contours(self) -> np.array:
        """The numpy array of contours"""
        contours = self.dataset.contours(
            self.representation,
            length=self.length,
            unique=self.unique,
            limit=self.limit,
        )

        # If another dimensionality is used, subsample the contours
        if self.dimensionality < contours.shape[1]:
            if self.representation == "cosine":
                contours = contours[:, : self.dimensionality]
            else:
                idx = np.linspace(0, contours.shape[1] - 1, self.dimensionality)
                contours = contours[:, idx.astype(int)]

        assert contours.shape[1] == self.dimensionality
        return contours

    @memoize
    def similarities(self, **kwargs) -> Union[bool, np.array]:
        """Get the pairwise similarity of all contours using the similarity
        metric of this condition."""
        if self.metric == "eucl":
            return self.eucl_similarities(**kwargs)
        elif self.metric == "dtw":
            return self.dtw_similarities(**kwargs)

    @catch_exceptions
    def eucl_similarities(self, **ignored_kws) -> np.array:
        """Pairwise euclidean distance"""
        return pdist(self.contours(), metric="euclidean")

    @serialize
    @catch_exceptions
    @log_start_end
    def dtw_similarities(self, **ignored_kws) -> np.array:
        """Pairwise dynamic time warping similarity. By default, these
        are serialized to the datasets HD5 store."""
        sim = cdist_dtw(self.contours(), **self.dtw_kws)
        return squareform(sim)

    @memoize
    @catch_exceptions
    def similarities_sample(
        self, limit: Optional[int] = SIM_DISTR_SAMPLE_SIZE
    ) -> np.array:
        """Return a sample of pairwise similarities

        Returns
        -------
        np.array
            A subsample of pairwise similarities
        """
        sim = self.similarities()
        limit = min(len(sim), limit)
        np.random.seed(42)
        sim = np.random.choice(sim, size=limit, replace=False)
        return sim

    @serialize
    @catch_exceptions
    @log_start_end
    def umap_embeddings(self, **ignored_kws) -> np.array:
        mapper = umap.UMAP(**self.umap_embed_kws)
        return mapper.fit_transform(self.contours())

    @serialize
    @catch_exceptions
    @log_start_end
    def umap_2d_embeddings(self, **ignored_kws) -> np.array:
        mapper = umap.UMAP(metric="precomputed", n_components=2)
        similarities = squareform(self.similarities())
        embeddings = mapper.fit_transform(similarities)
        return embeddings

    @memoize
    @serialize
    @catch_exceptions
    @log_start_end
    def kde_similarities(
        self, num_points: Optional[int] = 1000
    ) -> Union[bool, np.array]:
        sim = self.similarities_sample()
        kde = gaussian_kde(sim)
        margin = (sim.max() - sim.min()) * 0.05
        xs = np.linspace(sim.min() - margin, sim.max() + margin, num_points)
        ys = kde(xs)
        return np.c_[xs, ys]

    @memoize
    @serialize
    @catch_exceptions
    @log_start_end
    def unidip_dist_dip_test(
        self, num_tests: Optional[int] = 1000
    ) -> Tuple[float, float, Tuple[int, int]]:
        """Compute dist-dip test using the unidip package. This estimates the p-value
        using bootstrapping."""
        sim = self.similarities_sample()
        # _, (cdf_xs, cdf_ys, _, _, _, _) = dip_fn(sim)
        dip, p, (left, right) = diptst(sim, is_hist=False, numt=num_tests)
        return dict(dip=dip, p=p, left=left, right=right)

    @memoize
    @serialize
    @catch_exceptions
    @log_start_end
    def tableone_dist_dip_test(self) -> Tuple[float, float, Tuple[int, int]]:
        """Compute dist-dip test using the implementation from the tableone package.
        The p-value is computed by interpolating a table of precomputed values."""
        sim = self.similarities_sample()
        return tableone_dist_dip_test(sim)

    @memoize
    @serialize
    @catch_exceptions
    @log_start_end
    def umap_dist_dip_test(self) -> Tuple[float, float, Tuple[int, int]]:
        """Compute the dist-dip test on the pariwise distances of UMAP embeddings"""
        sim = pdist(self.umap_embeddings())
        limit = min(len(sim), SIM_DISTR_SAMPLE_SIZE)
        np.random.seed(42)
        data = np.random.choice(sim, size=limit, replace=False)
        return tableone_dist_dip_test(data)

    @create_file(output_dir=FIGURES_DIR, ext="pdf")
    @catch_exceptions
    @log_start_end
    def umap_sideplot(self, path: str):
        if self.metric == 'dtw':
            raise ValueError('Cannot create sideplot using a dtw metric')

        # UMAP
        mapper = umap.UMAP(random_state=0).fit(self.contours())
        gridpoints, inside = grid_around_points(
            mapper.embedding_, subdiv=10, margin=0.1
        )
        inv_contours = average_inv_contours(
            mapper.inverse_transform, gridpoints, num_samples=20, eps=0.3
        )

        # Inverse operations
        if self.representation == "cosine":
            inv_contours = idct(inv_contours, axis=1)
        elif self.representation in ["interval", "smooth_derivative"]:
            inv_contours = np.cumsum(inv_contours, axis=1)

        umap_plot_kws = dict()
        if "label" in self.df.columns:
            umap_plot_kws["labels"] = self.df["label"]

        # Plot
        fig = plt.figure(figsize=(16, 8), tight_layout=True)
        show_umap_sideplot(
            mapper, gridpoints, inside, inv_contours, umap_plot_kws=umap_plot_kws
        )
        fig.suptitle(repr(self)[1:-1], fontweight="bold")
        plt.savefig(path)
        plt.close()

    @create_file(output_dir=FIGURES_DIR, ext="pdf")
    # @catch_exceptions
    @log_start_end
    def umap_plot(self, path: str):
        kws = dict()
        if "label" in self.df.columns:
            kws["c"] = self.df["label"]
        embeddings = self.umap_2d_embeddings()
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        show_umap_plot(embeddings, scatter_kws=kws)
        fig.suptitle(repr(self)[1:-1], fontweight="bold")
        plt.savefig(path)
        plt.close()


def tableone_dist_dip_test(data) -> Tuple[float, float, Tuple[int, int]]:
    """Compute dist-dip test using the implementation from the tableone package.
    The p-value is computed by interpolating a table of precomputed values."""
    data = data[~np.isnan(data)]
    cdf_xs, cdf_ys = cum_distr(data)
    dip, (uni_xs, uni_ys) = dip_and_closest_unimodal_from_cdf(cdf_xs, cdf_ys)
    p = dip_pval_tabinterpol(dip, len(data))
    return dict(dip=dip, p=p)


if __name__ == "__main__":
    condition = Condition("clustered", "pitch_normalized", metric="eucl", limit=1000)
    # test = condition.umap_dist_dip_test(refresh_serialized=True)
    test = condition.umap_plot(refresh_file=True)
#     res = condition.tableone_dist_dip_test(refresh_serialized=True)
    ...
