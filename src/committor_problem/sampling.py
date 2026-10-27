from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
import torch
from project_runtime import make_generator, move_tensor, normal_noise, select_device

from .dynamics import langevin_step


@dataclass(frozen=True)
class MetadynamicsConfig:
    time_step: float = 1e-5
    deposit_every: int = 500
    n_hills: int = 2000
    hill_height: float = 5.0
    hill_width: float = 0.05
    n_samples: int = 210_000
    sample_stride: int = 1
    seed: int = 0
    log_every: int = 100
    burn_in_steps: int = 10_000
    device: str = "auto"


@dataclass(frozen=True)
class MetadynamicsResult:
    points: torch.Tensor
    hill_centers: torch.Tensor


def _bias_gradient(point, centers, height, width):
    displacement = point - centers
    hills = height * torch.exp(-0.5 * displacement.square().sum(dim=1) / width**2)
    return -(hills[:, None] * displacement).sum(dim=0) / width**2


def refresh_gaussian_coordinates(points, dimensions, standard_deviation, generator):
    dimensions = tuple(dimensions)
    refreshed = points.detach().clone()
    refreshed[:, dimensions] = standard_deviation * normal_noise(points[:, dimensions], generator)
    return refreshed


def metadynamics(
    potential: Callable[[torch.Tensor], torch.Tensor],
    initial_point: torch.Tensor,
    beta: float,
    config: MetadynamicsConfig | None = None,
    *,
    bias_dimensions: tuple[int, ...] = (0, 1),
) -> MetadynamicsResult:
    """Deposit Gaussian hills, then sample Langevin dynamics with the bias frozen."""
    if config is None:
        config = MetadynamicsConfig()

    point = move_tensor(initial_point.detach(), select_device(config.device)).clone()
    dimensions = torch.tensor(bias_dimensions, device=point.device)
    centers = point.new_empty((config.n_hills, len(bias_dimensions)))
    generator = make_generator(point.device, config.seed)

    def step(n_hills):
        nonlocal point
        bias_gradient = None
        if n_hills:
            bias_gradient = torch.zeros_like(point)
            bias_gradient[dimensions] = _bias_gradient(
                point[dimensions], centers[:n_hills],
                config.hill_height, config.hill_width,
            )
            bias_gradient = bias_gradient.unsqueeze(0)
        point = langevin_step(
            potential, point.unsqueeze(0), beta, config.time_step,
            generator, bias_gradient,
        ).squeeze(0)

    for hill in range(config.n_hills):
        for _ in range(config.deposit_every):
            step(hill)
        centers[hill] = point[dimensions]
        if config.log_every and (hill + 1) % config.log_every == 0:
            print(f"Metadynamics hills: {hill + 1}/{config.n_hills}", flush=True)

    for _ in range(config.burn_in_steps):
        step(config.n_hills)

    points = point.new_empty((config.n_samples, point.numel()))
    for sample in range(config.n_samples):
        for _ in range(config.sample_stride):
            step(config.n_hills)
        points[sample] = point
        if config.log_every and (sample + 1) % 20_000 == 0:
            print(f"Frozen-bias samples: {sample + 1}/{config.n_samples}", flush=True)

    return MetadynamicsResult(points, centers)


def frozen_bias_weights(
    points: torch.Tensor, hill_centers: torch.Tensor, beta: float,
    config: MetadynamicsConfig | None = None, *, bias_dimensions=(0, 1), batch_size=1024,
) -> torch.Tensor:
    if config is None:
        config = MetadynamicsConfig()
    centers = move_tensor(hill_centers.detach(), points.device, dtype=points.dtype)
    with torch.no_grad():
        log_weights = points.new_empty(len(points))
        for begin in range(0, len(points), batch_size):
            coordinates = points[begin:begin + batch_size, tuple(bias_dimensions)]
            bias = points.new_zeros(len(coordinates))
            for chunk in centers.split(512):
                squared_distance = (coordinates[:, None, :] - chunk[None, :, :]).square().sum(dim=-1)
                bias += config.hill_height * torch.exp(-0.5 * squared_distance / config.hill_width**2).sum(dim=1)
            log_weights[begin:begin + len(coordinates)] = beta * bias
        weights = torch.exp(log_weights - log_weights.max())
        return weights / weights.mean()


def delta_net(points: torch.Tensor, delta: float) -> torch.Tensor:
    coordinates = points.detach().cpu().double().numpy()
    if len(coordinates) == 0:
        return points.detach().clone()

    tree = cKDTree(coordinates)
    discarded = np.zeros(len(coordinates), dtype=bool)
    kept = []
    for index, point in enumerate(coordinates):
        if discarded[index]:
            continue
        kept.append(index)
        neighbors = np.asarray(tree.query_ball_point(point, delta), dtype=np.intp)
        displacement = coordinates[neighbors] - point
        close = np.einsum("ij,ij->i", displacement, displacement) < delta**2
        discarded[neighbors[close]] = True

    indices = torch.tensor(kept, device=points.device)
    return points.detach()[indices]
