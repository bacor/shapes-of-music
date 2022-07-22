from multiprocessing.sharedctypes import Value
import os
import logging
import json
import numpy as np
import pandas as pd
import h5py
from typing import List, Optional, Dict

from .config import CONTOUR_DIR, SERIALIZED_DIR, ALL_DATASETS, ALL_REPRESENTATIONS
from .representations import *
from .representations import contour_array


class InvalidDatasetException(Exception):
    """Raised when the dataset id is not known"""

    ...


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
    def __init__(self, dataset: str, refresh: Optional[bool] = False, log=False):
        if dataset not in ALL_DATASETS:
            raise InvalidDatasetException(f"Unknown dataset {dataset}")

        self.dataset = dataset
        self._df = None

        if log:
            log_fn = os.path.join(SERIALIZED_DIR, f"{self.dataset}.log")
            logging.basicConfig(
                filename=log_fn,
                filemode="a",
                format="%(levelname)s %(asctime)s %(message)s",
                datefmt="%d-%m-%y %H:%M:%S",
                level=logging.INFO,
            )

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

    ### HD5 store operations

    def exists(self, key: str) -> bool:
        """Test whether a key exists in the hd5 store

        Parameters
        ----------
        key : str
            The key

        Returns
        -------
        bool
            True if the key exists
        """
        with h5py.File(self.fn, "r") as file:
            exists = key in file
        return exists

    def store(self, path: str, obj: Dict, refresh: Optional[bool] = True):
        """Stores a dictionary (!) to the hd5 store

        Parameters
        ----------
        path : str
            The path/group where the dictionary is to be stored
        obj : Dict
            The dictionary
        refresh : Optional[bool], optional
            Whether to remove a possible existing object, by default True
        """
        if not isinstance(obj, dict):
            raise ValueError("Can only store dictionaries of numpy arrays")

        with h5py.File(self.fn, "r+") as file:
            for key, value in obj.items():
                subpath = f"{path}/{key}"
                if refresh and subpath in file.keys():
                    del file[subpath]
                if value is None:
                    value = np.nan
                if np.isscalar(value):
                    file.create_dataset(subpath, (), type(value), value)
                elif isinstance(value, np.ndarray):
                    file.create_dataset(subpath, value.shape, value.dtype, value)
                else:
                    raise ValueError(
                        f"Cannot store an object of type {type(value)}. "
                        "Only scalars and numpy arrays are supported."
                    )

    def load(self, path: str) -> dict:
        """Loads a dictionary stored to the hd5 store

        Parameters
        ----------
        path : str
            The path of the object in the data store

        Returns
        -------
        dict
            The loaded dictionary
        """
        output = dict()
        with h5py.File(self.fn, "r") as file:
            for key in file[path].keys():
                field = file[path][key]
                if field.shape == ():
                    # It is a scalar!
                    output[key] = field[()]
                else:
                    output[key] = field[:]
        return output

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

    ### Contours array

    def contours(self, representation: str, **subset_kwargs) -> np.array:
        """Return an array of contours in a certain representation. You can pass
        keyword arguments to select a particular subset of contours.

        Parameters
        ----------
        representation : str
            The name of the representation, e.g. 'pitch'

        Returns
        -------
        np.array
            A numpy array of shape (num_contours, num_samples)
        """
        if representation not in ALL_REPRESENTATIONS:
            raise Exception(f'Unknown representation "{representation}"')

        index = self.subset_index(**subset_kwargs)
        if len(index) == 0:
            return np.zeros((0, self.num_samples))
        else:
            with h5py.File(self.fn, "r+") as file:
                repr_fn = globals()[f"repr_{representation}"]
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

    def representation(self, *args, **kwargs):
        logging.warn('The method "representation()" is deprecated, use "contours()" instead') 
        return self.contours(*args, **kwargs)


# if __name__ == "__main__":
#     dataset = Dataset("markov", refresh=True)
#     ...
