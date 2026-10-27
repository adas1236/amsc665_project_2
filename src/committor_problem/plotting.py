import matplotlib.pyplot as plt
import numpy as np


def _get_axes(ax, figsize):
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def plot_loss(loss_epoch_average, ax=None):
    fig, ax = _get_axes(ax, (4, 3))
    ax.plot(np.arange(len(loss_epoch_average)), loss_epoch_average, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch-averaged loss")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_committor_histogram(probabilities, ax=None):
    fig, ax = _get_axes(ax, (6.2, 3.8))
    ax.hist(probabilities, bins=np.linspace(0, 1, 21), edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", label="$q=0.5$")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Estimated probability of reaching B first")
    ax.set_ylabel("Starting points")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_committor(
    x_grid,
    y_grid,
    q_grid,
    v_grid=None,
    *,
    potential_levels=None,
    boundaries=(),
    ax=None,
):
    fig, ax = _get_axes(ax, (4, 4))
    artist = ax.contourf(x_grid, y_grid, q_grid, levels=np.linspace(0, 1, 11))
    if v_grid is not None:
        ax.contour(x_grid, y_grid, v_grid, levels=potential_levels, colors="gray", linewidths=0.5)

    for coordinates, color in boundaries:
        ax.plot(coordinates[:, 0], coordinates[:, 1], linewidth=3, color=color)
    fig.colorbar(artist, ax=ax, label="Committor function", orientation="vertical")
    ax.set_aspect(1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig, ax
