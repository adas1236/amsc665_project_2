import matplotlib.pyplot as plt
import numpy as np


def plot_training(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5), squeeze=False)
    for ax, (family, models) in zip(axes[0], results.items()):
        for name, result in models.items():
            history = result["history"]
            epochs = np.arange(len(history["train_mse"]))
            line, = ax.plot(epochs, np.sqrt(history["train_mse"]), label=f"{name} train")
            keep = np.isfinite(history["validation_mse"])
            ax.plot(epochs[keep], np.sqrt(history["validation_mse"][keep]), linestyle="--", color=line.get_color(), label=f"{name} validation")
        ax.set_title(family)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    return fig, axes[0]


def plot_rmse_comparison(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5), squeeze=False)
    for ax, (family, models) in zip(axes[0], results.items()):
        names = list(models)
        positions = np.arange(len(names))
        for offset, split in ((-0.18, "train"), (0.18, "test")):
            values = [models[name]["metrics"][split]["RMSE"] for name in names]
            ax.bar(positions + offset, values, width=0.36, label=split.capitalize())
        ax.set_xticks(positions, names)
        ax.set_title(family)
        ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    return fig, axes[0]


def plot_predictions(results, datasets, n_examples=3):
    fig, axes = plt.subplots(len(results), n_examples, figsize=(4 * n_examples, 3 * len(results)), squeeze=False)
    for row, (family, models) in enumerate(results.items()):
        data = datasets[family].test
        grid = data.queries.cpu().numpy()
        for index in range(n_examples):
            ax = axes[row, index]
            ax.plot(grid, data.targets[index].cpu().numpy(), color="black", label="Chebyshev")
            for name, result in models.items():
                ax.plot(grid, result["predictions"][index], linestyle="--", label=name)
            ax.set_title(f"{family}, function {index + 1}")
            ax.set_xlabel("$x$")
            ax.set_ylabel("Committor")
            ax.legend()
    fig.tight_layout()
    return fig, axes
