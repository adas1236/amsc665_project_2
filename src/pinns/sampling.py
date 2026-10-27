from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
import torch
from project_runtime import move_tensor


@dataclass(frozen=True)
class TrainingPoints:
    initial: torch.Tensor
    boundary_left: torch.Tensor
    boundary_right: torch.Tensor
    interior: torch.Tensor

    def to(self, device, dtype=None):
        return TrainingPoints(*(move_tensor(data, device, dtype=dtype) for data in (
            self.initial, self.boundary_left, self.boundary_right, self.interior,
        )))


def sample_training_points(
    x_bounds, t_bounds, *, n_initial=50, n_boundary=50, n_interior=20_000,
    seed=0, dtype=torch.float32,
) -> TrainingPoints:
    xmin, xmax = x_bounds
    tmin, tmax = t_bounds
    rng = np.random.default_rng(seed)
    x_initial = rng.uniform(xmin, xmax, n_initial)
    t_boundary = rng.uniform(tmin, tmax, n_boundary)
    interior = qmc.scale(
        qmc.LatinHypercube(d=2, seed=seed).random(n_interior),
        [tmin, xmin], [tmax, xmax],
    )
    arrays = (
        np.column_stack((np.full(n_initial, tmin), x_initial)),
        np.column_stack((t_boundary, np.full(n_boundary, xmin))),
        np.column_stack((t_boundary, np.full(n_boundary, xmax))),
        interior,
    )
    return TrainingPoints(*(torch.as_tensor(array, dtype=dtype) for array in arrays))
