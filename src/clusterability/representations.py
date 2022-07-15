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
    "pitch",
    "pitch_centered",
    "pitch_normalized",
    "pitch_tonicized",
    "pitch_finalized",
    "cosine",
    "interval",
    "smooth_derivative",
]

__all__ = [f"repr_{name}" for name in REPRESENTATIONS]


def contour_array(df):
    """Extract a numpy array of shape (num_contours, num_samples) from a pandas
    DataFrame with contours. Timesteps are assumed to be numbered by sting-integers
    0, ..., num_steps"""
    num_samples = int(df.columns[-1]) + 1
    return df[[str(i) for i in range(num_samples)]].values


def repr_pitch(contours, **kwargs):
    """Return a numpy array of shape (num_contours, num_samples)"""
    return contours


def repr_pitch_centered(contours, **kwargs):
    """Center the contours so that every contour has mean pitch 0"""
    means = contours.mean(axis=1)
    return contours - means[:, np.newaxis]


def repr_pitch_normalized(contours, **kwargs):
    """Normalize contours to have maximum pitch 1 and minimum pitch 0.
    Flat contours are represented as a constant vector with value 0.5"""
    maxima = contours.max(axis=1)[:, np.newaxis]
    minima = contours.min(axis=1)[:, np.newaxis]
    normalized = (contours - minima) / (maxima - minima)
    # Fix completely flat contours
    flat_idx = np.where(maxima == minima)[0]
    normalized[flat_idx, :] = 0.5
    return normalized


def repr_pitch_tonicized(contours, tonic_krumhansl, tonic_mode, **kwargs):
    """Center a contour around the tonic of the melody. If the tonic_mode is not
    null, that value is used. Otherwise tonic_krumhansl is used."""
    tonics = tonic_mode
    indices_without_mode, = np.where(np.isnan(tonic_mode))
    tonics[indices_without_mode] = tonic_krumhansl[indices_without_mode]
    return contours - tonics[:, np.newaxis]


def repr_pitch_finalized(contours, final, **kwargs):
    """Center a contour around the final note of the melody."""
    return contours - final[:, np.newaxis]


def repr_cosine(contours, **kwargs):
    """Represent every contour as a cosine contour using the discrete
    cosine transform"""
    cosine_contours = dct(contours)
    cosine_contours[:, 0] = 0
    return cosine_contours


def repr_interval(contours, **kwargs):
    """Intervalic contour representation where every timestep represents the
    jump to the next timestep. A final 0 is appended to the contour representation
    to keep the dimensionality constant. This effectively means that the final
    note is extended for one timestep."""
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
    ys = np.convolve(window / window.sum(), padded_xs, mode="valid")
    return ys[int(window_len / 2) : -int(window_len / 2)]


def repr_smooth_derivative(contours, window_len=7, **kwargs):
    """Generalized intervalic representation where we take the gradient of the
    contour instead of intervals. Before taking the gradient, a hamming
    smoothing is applied to avoid a very irregular derivative due to the stepwise
    motion of the contour"""
    smooth = np.array([hamming_smoothing(c, window_len=window_len) for c in contours])
    return np.gradient(smooth, axis=1)
