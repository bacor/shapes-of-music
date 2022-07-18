import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def mean_contours(inverse_fn, points, num_samples=5, eps=.1):
    points = np.array(points, dtype=float)
    samples = np.tile(points, num_samples).reshape(len(points) * num_samples, 2)
    samples += np.random.normal(0, eps, size=samples.shape)
    contours = inverse_fn(samples)
    grouped = contours.reshape(len(points), num_samples, contours.shape[1])
    means = grouped.mean(axis=1)
    return means

def inset_plot(pos_x, pos_y, w=1, h=1, ax=None, **kwargs):
    if ax is None: ax = plt.gca()
    ax_ins = ax.inset_axes([pos_x - w/2, pos_y - h/2, w, h], transform=ax.transData, **kwargs)
    ax_ins.axis('off')
    return ax_ins
    
def get_grid_points(points, ax, xgrid=None, ygrid=None, radius=0.5):
    # Only use gridpoints inside the convex hull of the dataset
    if ax is None: ax = plt.gca()
    if xgrid is None: xgrid = ax.get_xticks()
    if ygrid is None: ygrid = ax.get_yticks()
    grid = np.array(list(product(xgrid, ygrid)))
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])
    is_inside = hull_path.contains_points(grid, radius=radius)
    return grid[is_inside]

def show_contour_grid(inverse_fn, points, ax=None, 
    num_samples=10, eps=0.5, radius=0.3, xgrid=None, ygrid=None, lw=1.5, c='k', **kwargs):
    if ax is None: ax = plt.gca()
    grid_points = get_grid_points(points, ax=ax, radius=radius, xgrid=xgrid, ygrid=ygrid)
    grid_contours = mean_contours(inverse_fn, grid_points, num_samples=num_samples, eps=eps)
    inset_axes = []
    for point, contour in zip(grid_points, grid_contours):
        props = dict()
        if len(inset_axes) > 0 : props['sharey'] = inset_axes[-1]
        ax_ins = inset_plot(*point, **props)
        ax_ins.plot(contour, 'w', lw=lw*3, zorder=-1)
        ax_ins.plot(contour, lw=lw, c=c, zorder=-1, **kwargs)
        inset_axes.append(ax_ins)
        ax.plot(*point, '+C3')