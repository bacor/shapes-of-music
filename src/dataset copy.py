from operator import le
import os 
import logging
import json
import numpy as np
import pandas as pd
import h5py

from scipy.spatial.distance import pdist, squareform
from tslearn.metrics import cdist_dtw, cdist_soft_dtw

from src.helpers import relpath, md5checksum, log_file
from src.representations import *
from src.representations import REPRESENTATIONS

_ROOT_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir)
_CONTOUR_DIR = os.path.join(_ROOT_DIR, 'contours')
_SERIALIZE_DIR = os.path.join(_ROOT_DIR, 'serialize')
_REPR_DIR = os.path.join(_SERIALIZE_DIR, 'representations')
_SIM_DIR = os.path.join(_SERIALIZE_DIR, 'similarities')

METRICS = ['eucl', 'dtw', 'soft_dtw']

class Dataset(object):
    
    def __init__(self, dataset, serialize_representations=False):
        self.dataset = dataset
        df_fn = os.path.join(_CONTOUR_DIR, f'{self.dataset}-contours.csv')
        self.df = pd.read_csv(df_fn, index_col=0)

        subset_fn = os.path.join(_CONTOUR_DIR, f'{dataset}-contours-subsets.json')
        with open(subset_fn, 'r') as handle:
            self.subsets = json.load(handle)

        self.fn = os.path.join(_SERIALIZE_DIR, f'{dataset}.h5')
        self.serialize_representations = serialize_representations
        self.serialized_repr_ = ['smooth_derivative', 'intervals']

        self.representations_ = {}
        self.similarities_ = {}

    def __repr__(self) -> str:
        return f'<Dataset {self.dataset}>'

    def log(self, message):
        logging.info(f'[{self.dataset}] {message}')

    def subset_index(self, subset_size=None, unique=False, length=None):
        # Precomputed random subsets
        if subset_size is None:
            index = np.arange(len(self.df))
        else:
            kind = 'unique' if unique else 'non-unique'
            index = self.subsets[kind].get(str(subset_size), False)
            if index is False: 
                return False
            else:
                index = np.array(index)
        
        # Filter by unit length
        if length is None:
            return index
        else:
            lengths = self.df['unit_length'].iloc[index]
            return index[lengths == length]

    def subset_df(self, subset_size=None, unique=False, length=None):
        index = self.subset_index(subset_size=subset_size, unique=unique, length=length)
        return self.df.iloc[index, :]

    ## Representations

    def load_representation(self, representation):
        with h5py.File(self.fn, 'a') as file:
            name = f'{self.dataset}/{representation}'
            if name in file.keys():
                self.log(f'Loading representation {representation} from {file}')
                return file[name][:]
            else: 
                return False

    def store_representation(self, representation, contours):
        with h5py.File(self.fn, 'a') as file:
            name = f'{self.dataset}/{representation}'
            if name in file.keys(): del file[name]
            file.create_dataset(name, contours.shape, contours.dtype, contours)
            self.log(f'Stored representation {representation} to {name}')
        
    def compute_representation(self, name):
        self.log(f'Computing representation {name}')
        repr_fn = globals()[f'repr_{name}']
        contours = repr_fn(self.df)
        return contours

    def representation(self, name, serialize=None, 
        subset_size=None, subset_unique=False, length=None):
        """"""
        if name not in REPRESENTATIONS:
            raise Exception(f'Unknown representation "{name}"')
        
        if serialize is None:
            serialize = True if name in self.serialized_repr_ else self.serialize_representations

        # Return representation from memory
        if name in self.representations_:
            contours = self.representations_[name]

        # Load or compute the representation
        else:
            contours = self.load_representation(name)
            if contours is False or serialize is False:
                contours = self.compute_representation(name)
            if serialize:
                self.store_representation(name, contours)
            self.representations_[name] = contours
        
        # Select a subset
        index = self.subset_index(subset_size=subset_size, unique=subset_unique, length=length)
        if index is not False:
            return contours[index, :]
        else:
            return False

    def precompute_representations(self):
        for name in REPRESENTATIONS:
            if self.serialize_representations or name in self.serialized_repr_:
                self.representation(name)

    ## Similarities

    def load_similarities(self, representation, metric, 
        subset_size=None, subset_unique=False, length=None):
        """"""
        unique = 'unique' if subset_unique else 'non_unique'
        filename = os.path.join(_SIM_DIR, f'{self.dataset}-{representation}-{metric}-{subset_size}-{unique}-{length}.txt.gz')
        if os.path.exists(filename):
            self.log(f'Loading {metric} similarity matrix from {relpath(filename)}')
            return np.loadtxt(filename)
        else:
            return False

    def store_similarities(self, similarities, representation, metric, 
        subset_size=None, subset_unique=False, length=None):
        """"""
        unique = 'unique' if subset_unique else 'non_unique'
        filename = os.path.join(_SIM_DIR, f'{self.dataset}-{representation}-{metric}-{subset_size}-{unique}-{length}.txt.gz')
        self.log(f'Storing {metric} similarity matrix to {relpath(filename)}')
        np.savetxt(filename, similarities)

    def compute_similarities(self, representation, metric, 
        subset_size=None, subset_unique=False, length=None,
        global_constraint='sakoe_chiba', sakoe_chiba_radius=20, gamma=0.01):
        """"""
        self.log(f'Computing {metric} similarity matrix for representation {representation}')
        self.log(f'   subset_size={subset_size}, subset_unique={subset_unique}')
        contours = self.representation(representation, subset_size=subset_size, subset_unique=subset_unique, length=length)
        if metric == 'eucl':
            similarities = pdist(contours, metric='euclidean')
        elif metric == 'dtw':
            similarities = cdist_dtw(contours, 
                global_constraint=global_constraint, sakoe_chiba_radius=sakoe_chiba_radius)
            similarities = squareform(similarities)
        elif metric == 'soft_dtw':
            similarities = cdist_soft_dtw(contours, gamma=gamma)
            raise NotImplemented
            # To do: squareform doesn't work here because of nonzero diagonals

        return similarities

    def similarities(self, representation, metric, serialize=True, 
        subset_size=None, subset_unique=False, length=None, **kwargs):
        if metric not in METRICS:
            raise Exception(f'Unknown metric "{metric}"')
        
        if metric.endswith('dtw') and representation.startswith('cosine'):
            raise Exception('Cannot use DTW similarity for cosine representations')

        key = (representation, metric, subset_size, subset_unique, length)

        # Return representation from memory
        if key in self.similarities_:
            similarities = self.similarities_[key]

        # Load or compute the representation
        else:
            similarities = self.load_similarities(representation, metric, 
                subset_size=subset_size, subset_unique=subset_unique, length=length)

            if similarities is False or serialize is False:
                similarities = self.compute_similarities(representation, metric, 
                    subset_size=subset_size, subset_unique=subset_unique, length=length, **kwargs)
                
                if serialize:
                    self.store_similarities(similarities, representation, metric, 
                        subset_size=subset_size, subset_unique=subset_unique, length=length)

            self.similarities_[key] = similarities
        
        return similarities
    
# if __name__ == '__main__':
#     dataset = Dataset('creighton-random')
#     contour = dataset.representation('pitch_normalized')
#     sim = dataset.similarities('pitch_normalized', 'dtw', subset_size=100)
#     print(contour)