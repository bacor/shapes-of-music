from genericpath import exists
from operator import le
import os 
import logging
import json
import numpy as np
import pandas as pd
import h5py

from scipy.spatial.distance import pdist, squareform
from tslearn.metrics import cdist_dtw, cdist_soft_dtw
from tslearn.metrics.dtw_variants import dtw

from src.helpers import relpath, md5checksum, log_file
from src.representations_h5 import *
from src.representations import REPRESENTATIONS

from scipy.stats import gaussian_kde

_ROOT_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir)
_CONTOUR_DIR = os.path.join(_ROOT_DIR, 'contours')
_SERIALIZE_DIR = os.path.join(_ROOT_DIR, 'serialize')

METRICS = ['eucl', 'dtw']

class Dataset(object):
    
    def __init__(self, dataset):
        self.dataset = dataset
        
        # Load subsets
        subset_fn = os.path.join(_CONTOUR_DIR, f'{dataset}-contours-subsets.json')
        with open(subset_fn, 'r') as handle:
            self.subsets = json.load(handle)

        # Load contours
        self.fn = os.path.join(_SERIALIZE_DIR, f'{dataset}.h5')
        with h5py.File(self.fn, 'a') as file:
            if 'contours' not in file.keys():
                self.log('Extracting contours and metadata columns.')
                df_fn = os.path.join(_CONTOUR_DIR, f'{self.dataset}-contours.csv')
                df = pd.read_csv(df_fn, index_col=0)
                self.num_samples = int(df.columns[-1]) + 1
                self.num_contours = len(df) 
                contours = df[[str(i) for i in range(self.num_samples)]].values
                file.create_dataset('contours', contours.shape, contours.dtype, contours)

                for col in ['tonic_krumhansl', 'tonic_mode', 'final', 'unit_length']:
                    column = df[col].values
                    file.create_dataset(f'meta/{col}', column.shape, column.dtype, column)

            else:
                self.num_contours = file['contours'].shape[0]
                self.num_samples = file['contours'].shape[1]

        self.similarities_ = {}

    def __repr__(self) -> str:
        return f'<Dataset {self.dataset}>'

    def log(self, message):
        logging.info(f'[{self.dataset}] {message}')

    def subset_index(self, subset_size=None, unique=False, length=None, limit=-1):
        # Precomputed random subsets
        if subset_size is None:
            index = np.arange(self.num_contours)
        else:
            kind = 'unique' if unique else 'non-unique'
            index = self.subsets[kind].get(str(subset_size), False)
            if index is False: 
                return False
            else:
                index = np.array(index)
        
        # Filter by unit length
        if length is None:
            return index[:limit]
        else:
            with h5py.File(self.fn, 'a') as file:
                lengths = file['meta/unit_length'][index]
                matches = index[lengths == length]
            return matches[:limit]

    def subset_name(self, subset_size=None, unique=False, length=None, limit=-1):
        size = 'all' if subset_size is None else subset_size
        unique = 'unique' if unique else 'non_unique'
        length = 'all' if length is None else length
        limit = 'none' if limit is -1 else limit
        return f'subset-{size}/{unique}/length-{length}/limit-{limit}'

    def subset_size(self, **subset_kwargs):
        return len(self.subset_index(**subset_kwargs))

    def representation(self, name, **subset_kwargs):
        """"""
        if name not in REPRESENTATIONS:
            raise Exception(f'Unknown representation "{name}"')
        
        index = self.subset_index(**subset_kwargs)
        if index is False or len(index) == 0:
            return np.zeros((0, self.num_samples))
        else:
            with h5py.File(self.fn, 'a') as file:
                self.log(f'Computing representation {name}')
                repr_fn = globals()[f'repr_{name}']
                contours = repr_fn(
                    file['contours'][index, :], 
                    final=file['meta/final'][index],
                    tonic_krumhansl=file['meta/tonic_krumhansl'][index],
                    tonic_mode=file['meta/tonic_mode'][index])[:]
        
        return contours
        
    def similarities(self, representation, metric, dtw_kwargs={}, **subset_kwargs):
        if metric not in METRICS:
            raise Exception(f'Unknown metric "{metric}"')
        
        if metric.endswith('dtw') and representation.startswith('cosine'):
            raise Exception('Cannot use DTW similarity for cosine representations')
        
        if metric == 'eucl':
            return self.eucl_similarities(representation, **subset_kwargs)

        elif metric == 'dtw':
            return self.dtw_similarities(representation, **dtw_kwargs, **subset_kwargs)

    def eucl_similarities(self, representation, **subset_kwargs):
        """"""
        index = self.subset_index(**subset_kwargs)
        if index is False or len(index) == 0:
            return np.zeros((0,))
        else:
            with h5py.File(self.fn, 'a') as file:
                repr_fn = globals()[f'repr_{representation}']
                contours = repr_fn(
                    file['contours'][index, :], 
                    final=file['meta/final'][index],
                    tonic_krumhansl=file['meta/tonic_krumhansl'][index],
                    tonic_mode=file['meta/tonic_mode'][index])
                similarities = pdist(contours, metric='euclidean')[:]
            return similarities

    def dtw_similarities(self, representation, refresh=False,
                         global_constraint='sakoe_chiba', sakoe_chiba_radius=20, 
                         **subset_kwargs):
        """"""
        with h5py.File(self.fn, 'a') as file:
            subset_name = self.subset_name(**subset_kwargs)
            name = f'dtw-similarity/{representation}/{subset_name}'
            if refresh or name not in file.keys():
                self.log(f'Computing dtw similarities and storing at {name}')
                contours = self.representation(representation, **subset_kwargs)
                if contours.shape[0] > 0:
                    similarities = cdist_dtw(contours, 
                                            global_constraint=global_constraint, 
                                            sakoe_chiba_radius=sakoe_chiba_radius)
                    similarities = squareform(similarities)
                else:
                    similarities = np.zeros((0,))

                if refresh: del file[name]
                file.create_dataset(name, similarities.shape, similarities.dtype, similarities)
            else:
                similarities = file[name][:]

        return similarities

    def similarity_kde(self, representation, metric, num_points=2000, refresh=False, 
        dtw_kwargs={}, **subset_kwargs):
        """"""
        with h5py.File(self.fn, 'a') as file:
            subset_name = self.subset_name(**subset_kwargs)
            name = f'kde/{representation}/{metric}/{subset_name}'
            if refresh or name not in file.keys():
                sim = self.similarities(representation, metric, 
                    dtw_kwargs=dtw_kwargs, **subset_kwargs)
                if len(sim) > 1:
                    kde = gaussian_kde(sim)
                    margin = (sim.max() - sim.min()) * 0.05
                    xs = np.linspace(sim.min() - margin, sim.max() + margin, num_points)
                    ys = kde(xs)
                else:
                    xs = np.array([])
                    ys = np.array([])

                distribution = np.c_[xs, ys]
                if refresh: del file[name]
                file.create_dataset(name, distribution.shape, distribution.dtype, distribution)
            else:
                distribution = file[name][:]
        
        return distribution

    def similarity_hist(self, *args, **kwargs):
        sim = self.similarities(*args, **kwargs)
        hist, bins = np.histogram(sim, bins='auto', density=True)
        return hist, bins

    def clean_similarity_kde(self):
        with h5py.File(self.fn, 'a') as file:
            if 'kde' in file.keys():
                del file['kde']

    # def hartigans_dip(self, *args, **kwargs, refresh=False):
    #     sim = self.similarities(*args, **kwargs)
    #     with h5py.File(self.fn, 'a') as file:
    #         subset_name = self.subset_name(**subset_kwargs)
    #         name = f'kde/{representation}/{metric}/{subset_name}'
        


