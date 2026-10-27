from dataclasses import dataclass

import numpy as np
import torch


class GaussianRandomField:
    def __init__(self, grid, length_scale=0.5, eigenvalue_threshold=1e-8):
        grid = np.asarray(grid, dtype=float)
        self.covariance = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / length_scale)**2)
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        keep = eigenvalues > eigenvalue_threshold
        self.factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])

    def sample(self, n_functions, rng):
        return rng.standard_normal((n_functions, self.factor.shape[1])) @ self.factor.T


@dataclass(frozen=True)
class DataConfig:
    n_train: int = 2000
    n_validation: int = 200
    n_test: int = 1000
    n_queries: int = 100
    n_evaluation: int = 257
    seed: int = 0


@dataclass(frozen=True)
class FunctionData:
    branch_inputs: torch.Tensor
    forces: torch.Tensor
    queries: torch.Tensor
    targets: torch.Tensor

    def inputs_for(self, kind):
        if kind == "branch":
            return self.branch_inputs
        if kind == "force":
            return self.forces


@dataclass(frozen=True)
class DatasetSplits:
    train: FunctionData
    validation: FunctionData
    test: FunctionData
    sensor_grid: np.ndarray


def make_splits(sampler, sensor_grid, solver, config=None):
    if config is None:
        config = DataConfig()
    grid = np.asarray(sensor_grid, dtype=float)
    rng = np.random.default_rng(config.seed)
    evaluation_grid = np.linspace(*solver.x_bounds, config.n_evaluation)
    splits = []
    for n_functions, training in ((config.n_train, True), (config.n_validation, False), (config.n_test, False)):
        branch, forces = (np.asarray(array, dtype=float) for array in sampler(n_functions, rng))
        queries = rng.uniform(*solver.x_bounds, size=(n_functions, config.n_queries)) if training else evaluation_grid
        targets = solver.from_samples(forces, grid, queries)
        splits.append(FunctionData(*(torch.as_tensor(array, dtype=torch.float32) for array in (branch, forces, queries, targets))))
    return DatasetSplits(*splits, grid.copy())


def save_splits(splits, path):
    arrays = {"sensor_grid": splits.sensor_grid}
    for name in ("train", "validation", "test"):
        data = getattr(splits, name)
        for field in ("branch_inputs", "forces", "queries", "targets"):
            arrays[f"{name}_{field}"] = getattr(data, field).cpu().numpy()
    np.savez_compressed(path, **arrays)
