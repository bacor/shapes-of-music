# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: 
# -------------------------------------------------------------------
""""""
import numpy as np
from scipy.fft import dct

REPRESENTATIONS = [
    'pitch',
    'pitch_centered', 
    'pitch_normalized',
    'pitch_tonicized',
    'pitch_finalized',
    'cosine',
    'cosine_centered',
    'cosine_normalized',
    'cosine_tonicized',
    'cosine_finalized',
    'interval',
    'smooth_derivative'
]

__all__ = [f'repr_{name}' for name in REPRESENTATIONS]

def contour_array(df):
    """Extract a numpy array of shape (num_contours, num_samples) from a pandas
    DataFrame with contours. Timesteps are assumed to be numbered by sting-integers
    0, ..., num_steps"""
    num_samples = int(df.columns[-1]) + 1
    return df[[str(i) for i in range(num_samples)]].values

def repr_pitch(df):
    """Return a numpy array of shape (num_contours, num_samples)"""
    contours = contour_array(df)
    return contours

def repr_pitch_centered(df):
    """Center the contours so that every contour has mean pitch 0"""
    contours = contour_array(df)
    means = contours.mean(axis=1)
    return contours - means[:, np.newaxis]

def repr_pitch_normalized(df):
    """Normalize contours to have maximum pitch 1 and minimum pitch 0.
    Flat contours are represented as a constant vector with value 0.5"""
    contours = contour_array(df)
    maxima = contours.max(axis=1)[:, np.newaxis]
    minima = contours.min(axis=1)[:, np.newaxis]
    normalized = (contours - minima) / (maxima - minima)
    # Fix completely flat contours
    flat_idx = np.where(maxima == minima)[0]
    normalized[flat_idx, :] = 0.5
    return normalized

def repr_pitch_tonicized(df):
    """Center a contour around the tonic of the melody. If the dataframe has a 
    nonempty tonic_mode column, these tonics are used. Otherwise tonic_krumhansl
    is used."""
    contours = contour_array(df)
    if df['tonic_mode'].isnull().all():
        return contours - df['tonic_krumhansl'][:, np.newaxis]
    else:
        return contours - df['tonic_mode'][:, np.newaxis]

def repr_pitch_finalized(df):
    """Center a contour around the final note of the melody."""
    contours = contour_array(df)
    return contours - df['final'][:, np.newaxis]

def repr_cosine(df):
    """Represent every contour as a cosine contour using the discrete 
    cosine transform"""
    contours = contour_array(df)
    return dct(contours)

def repr_cosine_centered(df):
    """Centered contours (with mean pitch 0) represented as cosine contours"""
    contours = repr_pitch_centered(df)
    return dct(contours)

def repr_cosine_normalized(df):
    """Normalized contours (minimum pitch 0, maximum pitch 1) represented
    as cosine contours"""
    contours = repr_pitch_normalized(df)
    return dct(contours)

def repr_cosine_tonicized(df):
    """Represent tonicized contours (tonic has pitch 0) as cosine contours"""
    contours = repr_pitch_tonicized(df)
    return dct(contours)

def repr_cosine_finalized(df):
    """Represent finalized contours (final has pitch 0) as cosine contours"""
    contours = repr_pitch_finalized(df)
    return dct(contours)

def repr_interval(df):
    """Intervalic contour representation where every timestep represents the
    jump to the next timestep. A final 0 is appended to the contour representation
    to keep the dimensionality constant. This effectively means that the final
    note is extended for one timestep."""
    contours = contour_array(df)
    intervals = contours[:, 1:] - contours[:, :-1]
    zeros = np.zeros((contours.shape[0], 1))
    return np.append(zeros, intervals, axis=1) 

def hamming_smoothing(xs, window_len=7):
    """Smooth an array of numbers using a Hamming window"""
    # https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth.html
    assert window_len % 2 == 1
    prefix = [xs[0]] * (window_len - 1)
    suffix = [xs[-1]] * (window_len - 1)
    padded_xs = np.concatenate([prefix, xs, suffix])
    window = np.hamming(window_len)
    ys = np.convolve(window/window.sum(), padded_xs, mode='valid')
    return ys[int(window_len/2):-int(window_len/2)]

def repr_smooth_derivative(df, window_len=7):
    """Generalized intervalic representation where we take the gradient of the
    contour instead of intervals. Before taking the gradient, a hamming 
    smoothing is applied to avoid a very irregular derivative due to the stepwise
    motion of the contour"""
    contours = contour_array(df)
    smooth = np.array([hamming_smoothing(c, window_len=window_len) for c in contours])
    return np.gradient(smooth, axis=1)

# def generate_representations(prefix):
#     """Generate all representations for one dataset"""
#     contours_fn = os.path.join(_CONTOUR_DIR, f'{prefix}-contours.csv')
#     logging.info(f'Extracting representations for {prefix}')
#     df = pd.read_csv(contours_fn, index_col=0)

#     for representation in _REPRESENTATIONS:
#         repr_fn = globals()[f'repr_{representation}']
#         contours = repr_fn(df)
#         filename = os.path.join(_REPR_DIR, f'{prefix}-{representation}.txt.gz')
#         np.savetxt(filename, contours)
#         logging.info(f'  > Stored representation {representation} to {relpath(filename)}')

# def generate_all_representations():
#     dataset_ids = [
#         'erk', 'boehme', 'creighton',
#         'han', 'natmin', 'shanxi',
#         'liber-antiphons', 'liber-responsories', 'liber-alleluias']

#     log_fn = os.path.join(_REPR_DIR, f'representations.log')
#     logging.basicConfig(filename=log_fn, **_LOGGING_OPTIONS)
#     logging.info(f'Generating contour representations')
#     logging.info(dataset_ids)
#     for dataset_id in dataset_ids:
#         for kind in ['phrase', 'random']:
#             generate_representations(f'{dataset_id}-{kind}')

#     for genre in ['antiphon', 'responsory']:
#         for segmentation in ['neumes', 'syllables', 'words', 'poisson-3', 'poisson-5', 'poisson-7']:
#             generate_representations(f'cantus-{genre}-{segmentation}')

# if __name__ == '__main__':
#     generate_all_representations()

