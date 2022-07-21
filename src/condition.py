import os
import logging
from functools import wraps
from typing import Callable, List, Optional, Dict, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from tslearn.metrics import cdist_dtw
from unidip.dip import dip_fn, diptst

from .config import (
    FIGURES_DIR,
    ALL_METRICS,
    CONDITIONS_PER_DATASET,
    METRIC_LIMITS,
    MIN_CONTOUR_COUNT,
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
        self.log(f'Start running {name}')
        output = func(self, *args, **kwargs)
        self.log(f'> Done running {name}')
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

        if metric not in ALL_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if not validate_condition(self.as_dict()):
            raise InvalidConditionException(
                "The given combination of options is not allowed. "
                "See the variable `INVALID_COMBINATIONS` in `config.py` for "
                "all invalid combinations."
            )
        
        if log:
            self.log(
                f'{repr(self)[1:-1]}'
            )

        # Store DTW options
        self.dtw_kws = dict(
            global_constraint="sakoe_chiba", sakoe_chiba_radius=20, n_jobs=4
        )
        self.dtw_kws.update(dtw_kws)

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
            f"{self.representation}-{self.metric}",
            f"length_{length}",
            f"limit_{limit}-subset_{subset}-dim_{dim}",
        ]
        return parts

    def log(self, message: str):
        self.dataset.log(message)

    @memoize
    def contours(self) -> np.array:
        """The numpy array of contours"""
        contours = self.dataset.contours(
            self.representation,
            length=self.length,
            unique=self.unique,
            limit=self.limit,
        )
        if contours.shape[0] < MIN_CONTOUR_COUNT:
            raise TooFewContoursException(
                "Not enough contours were found for this condition. Only "
                f"{contours.shape[0]} contours were found, but at least "
                f"{MIN_CONTOUR_COUNT} are required."
            )

        # TODO : dimensionality
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
        self.log(f"Computing DTW similiarities...")
        sim = cdist_dtw(self.contours(), **self.dtw_kws)
        self.log(f"> done.")
        return squareform(sim)
    

    @memoize
    @serialize
    @catch_exceptions
    @log_start_end
    def kde_similarities(
        self, num_points: Optional[int] = 2000
    ) -> Union[bool, np.array]:
        self.log(f"Computing kernel density estimate...")
        sim = self.similarities()
        kde = gaussian_kde(sim)
        margin = (sim.max() - sim.min()) * 0.05
        xs = np.linspace(sim.min() - margin, sim.max() + margin, num_points)
        ys = kde(xs)
        self.log(f"> done.")
        return np.c_[xs, ys]

    @memoize
    @serialize
    @catch_exceptions
    @log_start_end
    def dist_dip_test(self, num_tests: Optional[int] = 1000):
        self.log(f"Computing dist-dip test")
        sim = self.similarities()
        _, (cdf, xs, _, _, _, _) = dip_fn(sim)
        dip, pval, (left, right) = diptst(sim, is_hist=False, numt=num_tests)
        self.log(f"> done.")
        return dict(
            dip=np.array([dip]),
            pval=np.array([pval]),
            left=np.array([left]),
            right=np.array([right]),
            xs=xs,
            cdf=cdf,
        )

    @create_file(output_dir=FIGURES_DIR, ext="pdf")
    def umap_plot(self, path):
        if self.metric == "eucl":
            self.create_umap_sideplot(path)
        elif self.metric == "dtw":
            self.create_umap_plot(path)

    @catch_exceptions
    @log_start_end
    def create_umap_sideplot(self, path: str):
        import matplotlib.pyplot as plt
        import umap
        import umap.plot
        from scipy.fft import idct
        from .visualize import (
            average_inv_contours,
            grid_around_points,
            show_side_plot,
        )

        self.log("Creating UMAP side plot...")

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

        # Plot
        fig = plt.figure(figsize=(16, 8), tight_layout=True)
        show_side_plot(mapper, gridpoints, inside, inv_contours)
        fig.suptitle(repr(self)[1:-1], fontweight="bold")
        plt.savefig(path)
        plt.close()

    @catch_exceptions
    @log_start_end
    def create_umap_plot(self, path: str):
        import matplotlib.pyplot as plt
        import umap
        import umap.plot

        self.log("Creating simple UMAP plot...")
        sims = squareform(self.similarities())
        mapper = umap.UMAP(random_state=0, metric="precomputed")
        mapper.fit(sims)

        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        umap.plot.points(mapper, ax=plt.gca())
        fig.suptitle(repr(self)[1:-1], fontweight="bold")
        plt.savefig(path)
        plt.close()

if __name__ == "__main__":
    condition = Condition("markov", "cosine", metric="dtw", limit=10)
    ...
