# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

def cm2inch(*args):
    return list(map(lambda x: x/2.54, args))

def title(text, ha='left', x=0, fontweight='bold', ax=None, **kwargs):
    if ax is None: ax = plt.gca()
    ax.set_title(text, ha=ha, x=x, fontweight=fontweight, **kwargs)

def show_num_contours(num_contours, ax, y=1.06):
    plt.text(1, y, f'N={num_contours}', transform=ax.transAxes, 
         va='bottom', ha='right', size=6, fontstyle='italic', color='0.6')

def normalized_contours(df: pd.DataFrame, normalize=True):
    """Extract a Numpy array of normalized pitch contours from a dataframe
    with pitch contours"""
    start_index = list(df.columns).index('0')
    pitches = df.iloc[:, start_index:].values
    means = pitches.mean(axis=1)
    normalized_pitches = pitches - means[:, np.newaxis]
    if normalize:
        return normalized_pitches
    else:
        return pitches
    
def load_datasets(
    dataset_ids=[
        'erk',
        'boehme',
        'creighton',
        'han', 
        'natmin',
        'shanxi',
        # 'essen-europe',
        # 'essen-china',
        # 'essen-europe-china',
        'liber-antiphons',
        'liber-responsories',
        'liber-alleluias'
    ], 
    include_random=True, dir='../contours', normalize=True):

    dfs = {}
    contours = {}
    for dataset_id in dataset_ids:
        dfs[dataset_id] = pd.read_csv(f'{dir}/{dataset_id}-phrase-contours.csv', index_col=0)
        dfs[f'{dataset_id}-random'] = pd.read_csv(f'{dir}/{dataset_id}-random-contours.csv', index_col=0)
        contours[dataset_id] = normalized_contours(dfs[dataset_id], normalize=normalize)
        contours[f'{dataset_id}-random'] = normalized_contours(dfs[f'{dataset_id}-random'],
                                                               normalize=normalize)
    return dfs, contours


def format_pval(p, alpha=0.05, tol=1e-8):
    if np.isnan(p):
        return 'NA'
    elif np.isclose(p, 0, tol):
        return '0'
    elif p >= 0.1:
        out = f'{p:.1f}'
        if out == '1.0':
            out  = '1'
        return out
    elif p >= 0.001:
        return f'{p:.3f}'
    else:
        p_str = f'{p:.1e}'
        return p_str.replace('e-0', 'e-')

def get_pval_cmap(
        alpha=0.05,
        vmin=1e-8, vmax=1, 
        min_gray=0.03, max_gray=0.5, 
        min_cmap=0, max_cmap=0.45,
        cmap='viridis_r'
    ):
    split = (np.log10(alpha) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    colors1 = plt.cm.gray_r(np.linspace(min_gray, max_gray, int(1000*(1-split))))
    cmap = plt.get_cmap(cmap)
    colors2 = cmap(np.linspace(min_cmap, max_cmap, int(1000*split)))
    colors = np.vstack((colors1, colors2))[::-1]
    return mcolors.LinearSegmentedColormap.from_list('pval_colormap', colors)
