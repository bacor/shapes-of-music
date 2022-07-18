import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import umap.plot

from itertools import product
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform
from scipy.fft import idct
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from ..config import FIGURES_DIR
from ..helpers import relpath
from .dataset import CONDITIONS, PRECOMPUTED_CONDITIONS, PRECOMPUTED_LENGTHS, Dataset


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
    _inset_plot_kws = dict(lw=1.5, c="k")
    _inset_plot_kws.update(inset_plot_kws)
    _inset_bg_plot_kws = dict(lw=4 * _inset_plot_kws["lw"], c="w", alpha=0.7)
    _inset_bg_plot_kws.update(inset_bg_plot_kws)
    _marker_kws = dict(marker="+", c="C3", s=30, lw=1)
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


def show_side_plot(
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
    _marker_kws = dict(c="k", marker="x", s=25, lw=1)
    _marker_kws.update(marker_kws)
    _outside_marker_kws = dict(**_marker_kws)
    _outside_marker_kws.update(dict(alpha=0.1))
    _outside_marker_kws.update(outside_marker_kws)
    _plot_kws = dict(c="k", ls="-")
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


def create_dtw_plot(dataset_id, limit: int = 500, refresh=False):
    output_dir = os.path.join(FIGURES_DIR, "fig-umap-sideplot", dataset_id)

    for representation, metrics in CONDITIONS.items():
        if "dtw" not in metrics:
            continue
        for length in PRECOMPUTED_LENGTHS:

            # Output file
            plot_fn = os.path.join(
                output_dir,
                representation,
                f"{dataset_id}-{representation}-DTW-length{length}.pdf",
            )
            directory = os.path.dirname(plot_fn)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.exists(plot_fn) and not refresh:
                print(f"Skipping {relpath(plot_fn)}")
                continue

            try:
                dataset = Dataset(dataset_id)
                sims = dataset.similarities(
                    representation,
                    metric="dtw",
                    limit=limit,
                    length=length,
                    unique=False,
                )

                # UMAP
                mapper = umap.UMAP(random_state=0, metric="precomputed")
                mapper.fit(squareform(sims))

                # Plot
                fig = plt.figure(figsize=(16, 8), tight_layout=True)
                umap.plot.points(mapper)
                title = (
                    f"{dataset_id}, representation={representation}, length={length}\n"
                )
                fig.suptitle(title, fontweight="bold")
                plt.savefig(plot_fn)

            except Exception as e:
                print(f"An error occured: {e}")
                print(
                    f"> dataset={dataset_id}, repres={representation}, length={length}"
                )


def create_side_plot(dataset_id, limit: int = 5000, refresh=False):
    output_dir = os.path.join(FIGURES_DIR, "fig-umap-sideplot", dataset_id)
    for representation, length in product(PRECOMPUTED_CONDITIONS, PRECOMPUTED_LENGTHS):

        # Output file
        plot_fn = os.path.join(
            output_dir,
            representation,
            f"{dataset_id}-{representation}-length{length}.pdf",
        )
        directory = os.path.dirname(plot_fn)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(plot_fn) and not refresh:
            print(f"Skipping {relpath(plot_fn)}")
            continue

        try:
            # Load dataset
            dataset = Dataset(dataset_id)
            contours = dataset.representation(
                representation, limit=limit, length=length, unique=False
            )

            # UMAP
            mapper = umap.UMAP(random_state=0).fit(contours)
            gridpoints, inside = grid_around_points(
                mapper.embedding_, subdiv=10, margin=0.1
            )
            # inv_contours = mapper.inverse_transform(gridpoints)
            inv_contours = average_inv_contours(
                mapper.inverse_transform, gridpoints, num_samples=20, eps=0.3
            )

            # for cosine contours show existing
            if representation == "cosine":
                inv_contours = idct(inv_contours, axis=0)

            # Plot
            fig = plt.figure(figsize=(16, 8), tight_layout=True)
            show_side_plot(mapper, gridpoints, inside, inv_contours)
            title = f"{dataset_id}, representation={representation}, length={length}\n"
            fig.suptitle(title, fontweight="bold")
            plt.savefig(plot_fn)

        except Exception as e:
            print(f"An error occured: {e}")
            print(f"> dataset={dataset_id}, repres={representation}, length={length}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute all similarity scores and so on"
    )
    parser.add_argument("plot", type=str, help="Plot type")
    parser.add_argument("dataset", type=str, help="ID of the dataset to visualize")
    args = parser.parse_args()

    if args.plot == "sideplot":
        create_side_plot(args.dataset)
    elif args.plot == "dtwplot":
        create_dtw_plot(args.dataset)
    else:
        raise ValueError("Unknown plot type")


if __name__ == "__main__":
    main()
