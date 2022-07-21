import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
from itertools import product
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec


def average_inv_contours(inverse_fn, points, num_samples=5, eps=0.1):
    points = np.array(points, dtype=float)
    samples = np.tile(points, num_samples).reshape(len(points) * num_samples, 2)
    samples += np.random.normal(0, eps, size=samples.shape)
    contours = inverse_fn(samples)
    grouped = contours.reshape(len(points), num_samples, contours.shape[1])
    means = grouped.mean(axis=1)
    return means


def grid_around_points(points, subdiv: int = 10, margin: float = 0.1):
    # Compute convex hull
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    # Grid in box around dataset
    xmin, ymin = hull.min_bound
    xmax, ymax = hull.max_bound
    width = xmax - xmin
    height = ymax - ymin
    xmargin = margin * height
    ymargin = margin * width
    xgrid = np.linspace(xmin - xmargin, xmax + xmargin, subdiv)
    ygrid = np.linspace(ymin - ymargin, ymax + ymargin, subdiv)
    gridpoints = np.array(list(product(xgrid, ygrid)))

    # Determine gridpoints inside/outside the convex hull
    radius = margin * min(height, width)
    inside = hull_path.contains_points(gridpoints, radius=radius)

    return gridpoints, inside


def set_spine_color(color, ax=None):
    if ax is None:
        ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color(color)


def set_spine_alpha(alpha, ax=None):
    if ax is None:
        ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_alpha(alpha)


def show_inset_plot(
    points_or_mapper,
    gridpoints,
    inside,
    inv_contours,
    bg=True,
    marker=True,
    inset_height=1,
    inset_width=1,
    scatter_kws={},
    umap_plot_kws={},
    inset_plot_kws={},
    inset_bg_plot_kws={},
    outside_plot_kws={},
    marker_kws={},
    ax=None,
):
    _scatter_kws = dict(s=0.1, cmap="Spectral")
    _scatter_kws.update(scatter_kws)
    _umap_plot_kws = dict()
    _umap_plot_kws.update(umap_plot_kws)
    _inset_plot_kws = dict(lw=1.5, color="k")
    _inset_plot_kws.update(inset_plot_kws)
    _inset_bg_plot_kws = dict(lw=4 * _inset_plot_kws["lw"], color="w", alpha=0.7)
    _inset_bg_plot_kws.update(inset_bg_plot_kws)
    _marker_kws = dict(marker="+", color="C3", s=30, lw=1)
    _marker_kws.update(marker_kws)
    _outside_plot_kws = dict(**_inset_plot_kws)
    _outside_plot_kws.update(dict(alpha=0.2))
    _outside_plot_kws.update(outside_plot_kws)
    if ax is None:
        ax = plt.gca()

    # Scatter plot or umap plot
    ax.set(xticks=[], yticks=[])
    if isinstance(points_or_mapper, umap.UMAP):
        umap.plot.points(points_or_mapper, ax=ax, **_umap_plot_kws)
    else:
        ax.scatter(points_or_mapper[:, 0], points_or_mapper[:, 1], **_scatter_kws)
    ax.set_axis_off()

    # Plot inset axes
    inset_axes = []
    for point, is_inside, contour in zip(gridpoints, inside, inv_contours):
        props = dict()
        if len(inset_axes) > 0:
            props["sharey"] = inset_axes[-1]

        # Inset axis
        xpos = point[0] - inset_width / 2
        ypos = point[1] - inset_height / 2
        ax_ins = ax.inset_axes(
            [xpos, ypos, inset_width, inset_height], transform=ax.transData, **props
        )
        ax_ins.set(xticks=[], yticks=[])
        ax_ins.patch.set_alpha(0)
        set_spine_alpha(0.25, ax=ax_ins)
        inset_axes.append(ax_ins)

        # Plot the contour
        if bg:
            ax_ins.plot(contour, zorder=-1, **_inset_bg_plot_kws)
        if marker:
            ax.scatter(*point, **_marker_kws)
        if is_inside:
            ax_ins.plot(contour, zorder=-1, **_inset_plot_kws)
        else:
            ax_ins.plot(contour, zorder=-1, **_outside_plot_kws)


def show_umap_sideplot(
    points_or_mapper,
    gridpoints,
    inside,
    inv_contours,
    fig=None,
    scatter_kws={},
    umap_plot_kws={},
    marker_kws={},
    outside_marker_kws={},
    plot_kws={},
    outside_plot_kws={},
):

    # Default styles
    _scatter_kws = dict(s=0.1, cmap="Spectral")
    _scatter_kws.update(scatter_kws)
    _umap_plot_kws = dict()
    _umap_plot_kws.update(umap_plot_kws)
    _marker_kws = dict(color="k", marker="x", s=25, lw=1)
    _marker_kws.update(marker_kws)
    _outside_marker_kws = dict(**_marker_kws)
    _outside_marker_kws.update(dict(alpha=0.1))
    _outside_marker_kws.update(outside_marker_kws)
    _plot_kws = dict(color="k", ls="-")
    _plot_kws.update(plot_kws)
    _outside_plot_kws = dict(**_plot_kws)
    _outside_plot_kws.update(dict(alpha=0.1))
    _outside_plot_kws.update(outside_plot_kws)

    # Set up figure axes
    if fig is None:
        fig = plt.gcf()
    subdiv = int(np.sqrt(len(gridpoints)))
    gs = GridSpec(subdiv, 2 * subdiv, fig)
    scatter_ax = fig.add_subplot(gs[:, :subdiv])
    digit_axes = np.zeros((subdiv, subdiv), dtype=object)
    for i in range(subdiv):
        for j in range(subdiv):
            props = dict()
            if i > 0 and j > 0:
                props["sharey"] = digit_axes[0, 0]
            digit_axes[i, j] = fig.add_subplot(gs[i, subdiv + j], **props)

    # Scatter plot or umap plot
    scatter_ax.set(xticks=[], yticks=[])
    if isinstance(points_or_mapper, umap.UMAP):
        umap.plot.points(points_or_mapper, ax=scatter_ax, **_umap_plot_kws)
    else:
        scatter_ax.scatter(
            points_or_mapper[:, 0], points_or_mapper[:, 1], **_scatter_kws
        )

    # Show grid points
    scatter_ax.scatter(gridpoints[inside, 0], gridpoints[inside, 1], **_marker_kws)
    scatter_ax.scatter(
        gridpoints[inside == False, 0],
        gridpoints[inside == False, 1],
        **_outside_marker_kws,
    )

    # Reshape everything to subdiv x subdiv grid
    # gridpoints = gridpoints.reshape(subdiv, subdiv, 2)
    inside = inside.reshape(subdiv, subdiv)
    inv_contours = inv_contours.reshape(subdiv, subdiv, -1)

    # Plot contours in all subplots of the side plot
    for i in range(subdiv):
        for j in range(subdiv):
            # gridpoints are indexed from the bottom left, subplots from top left
            ax = digit_axes[subdiv - 1 - j, i]
            ax.set(xticks=[], yticks=[])
            if not inside[i, j]:
                set_spine_alpha(0.1, ax=ax)
                ax.plot(inv_contours[i, j], **_outside_plot_kws)
            else:
                ax.plot(inv_contours[i, j], **_plot_kws)


def show_umap_plot(points_or_mapper, ax=None, umap_plot_kws={}, scatter_kws={}):
    if ax is None:
        ax = plt.gca()
    _umap_plot_kws = dict()
    _umap_plot_kws.update(umap_plot_kws)
    _scatter_kws = dict(s=0.1, cmap="Spectral")
    _scatter_kws.update(scatter_kws)

    if isinstance(points_or_mapper, umap.UMAP):
        umap.plot.points(points_or_mapper, ax=ax, **_umap_plot_kws)
    else:
        ax.scatter(points_or_mapper[:, 0], points_or_mapper[:, 1], **_scatter_kws)
