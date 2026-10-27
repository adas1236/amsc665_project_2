import matplotlib.pyplot as plt
import numpy as np


def plot_losses(history):
    fig, ax = plt.subplots(figsize=(4, 3))
    for name, label in (("pde", "PDE"), ("initial", "IC"), ("boundary", "BC")):
        ax.plot(np.arange(len(history[name])), history[name], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_heatmaps(t, x, reference, predicted):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    limit = max(np.max(np.abs(reference)), np.max(np.abs(predicted)), 1e-12)
    levels = np.linspace(-limit, limit, 101)
    for ax, values, title in zip(axes[:2], (reference, predicted), ("Spectral", "PINN")):
        artist = ax.contourf(t, x, values, levels=levels, cmap="RdBu_r")
        fig.colorbar(artist, ax=ax, label="$u(t,x)$")
        ax.set_title(title)
    artist = axes[2].contourf(t, x, np.abs(predicted - reference), levels=100)
    fig.colorbar(artist, ax=axes[2], label="Absolute error")
    axes[2].set_title("Absolute error")
    for ax in axes:
        ax.set_xlabel("$t$")
    axes[0].set_ylabel("$x$")
    fig.tight_layout()
    return fig, axes


def _time_slice(t, values, query):
    t = np.asarray(t)
    right = int(np.searchsorted(t, query))
    if right == 0:
        return values[:, 0]
    weight = (query - t[right - 1]) / (t[right] - t[right - 1])
    return (1 - weight) * values[:, right - 1] + weight * values[:, right]


def plot_snapshots(t, x, reference, predicted, times=(0.1, 0.9)):
    fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 3.5), squeeze=False)
    for ax, query in zip(axes[0], times):
        ax.plot(x, _time_slice(t, reference, query), label="Spectral")
        ax.plot(x, _time_slice(t, predicted, query), linestyle="--", label="PINN")
        ax.set_title(f"$t={query:g}$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.legend()
    fig.tight_layout()
    return fig, axes[0]
