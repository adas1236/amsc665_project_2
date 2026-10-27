from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

from matplotlib.figure import Figure
import numpy as np
import torch
from project_runtime import make_generator, move_tensor, select_device

from .dynamics import langevin_step


@dataclass(frozen=True)
class ShootingConfig:
    trajectories_per_point: int = 200
    time_step: float = 1e-5
    max_steps: int = 1_000_000
    batch_size: int = 1000
    seed: int = 1
    log_every: int = 10_000
    device: str = "auto"


@dataclass(frozen=True)
class CommittorAnalysisResult:
    hits_a: np.ndarray
    hits_b: np.ndarray
    unresolved: np.ndarray

    @property
    def probabilities(self) -> np.ndarray:
        return self.hits_b / (self.hits_a + self.hits_b)


def extract_isocommittor(x_grid, y_grid, q_grid, level=0.5) -> list[np.ndarray]:
    values = np.ma.masked_invalid(q_grid)

    figure = Figure()
    contours = figure.subplots().contour(x_grid, y_grid, values, levels=[level])
    curves = [
        segment.copy() for segment in contours.allsegs[0]
        if len(segment) >= 2 and np.isfinite(segment).all()
        and np.linalg.norm(np.diff(segment, axis=0), axis=1).sum() > 0
    ]
    figure.clear()
    return curves


def sample_isocommittor(
    curves: Sequence[np.ndarray],
    potential: Callable[[torch.Tensor], torch.Tensor],
    beta: float,
    n_points: int,
    dimension: int,
    *,
    n_candidates: int = 10_000,
    seed: int = 1,
) -> torch.Tensor:
    polylines = [np.asarray(curve, dtype=float) for curve in curves]

    starts = np.concatenate([curve[:-1] for curve in polylines])
    directions = np.concatenate([np.diff(curve, axis=0) for curve in polylines])
    lengths = np.linalg.norm(directions, axis=1)
    nonzero = lengths > 0
    starts, directions, lengths = starts[nonzero], directions[nonzero], lengths[nonzero]
    cumulative = np.cumsum(lengths)
    locations = (np.arange(n_candidates) + 0.5) * cumulative[-1] / n_candidates
    segments = np.searchsorted(cumulative, locations, side="right")
    preceding = np.concatenate(([0.0], cumulative[:-1]))
    fractions = (locations - preceding[segments]) / lengths[segments]
    xy = starts[segments] + fractions[:, None] * directions[segments]

    candidates = torch.zeros((n_candidates, dimension), dtype=torch.float64)
    candidates[:, :2] = torch.from_numpy(xy)
    with torch.no_grad():
        energies = potential(candidates).detach().cpu().numpy().reshape(-1)
    log_weights = -beta * energies
    weights = np.exp(log_weights - log_weights.max())
    probabilities = weights / weights.sum()
    indices = np.random.default_rng(seed).choice(n_candidates, size=n_points, p=probabilities)
    return candidates[indices]


def committor_analysis(
    potential: Callable[[torch.Tensor], torch.Tensor],
    starting_points: torch.Tensor,
    beta: float,
    in_a: Callable[[torch.Tensor], torch.Tensor],
    in_b: Callable[[torch.Tensor], torch.Tensor],
    config: ShootingConfig | None = None,
) -> CommittorAnalysisResult:
    if config is None:
        config = ShootingConfig()

    n_points = len(starting_points)
    device = select_device(config.device)
    starting_points = move_tensor(starting_points.detach(), device)
    generator = make_generator(device, config.seed)
    hits_a = torch.zeros(n_points, dtype=torch.int64, device=device)
    hits_b = torch.zeros_like(hits_a)
    unresolved = torch.zeros_like(hits_a)
    total = n_points * config.trajectories_per_point

    def absorb(points, owners):
        with torch.no_grad():
            a_mask = in_a(points).reshape(-1)
            b_mask = in_b(points).reshape(-1)
            for mask, counts in ((a_mask, hits_a), (b_mask, hits_b)):
                indices = owners[mask]
                counts.scatter_add_(0, indices, torch.ones_like(indices))
            remaining = ~(a_mask | b_mask)
            return points[remaining], owners[remaining]

    for begin in range(0, total, config.batch_size):
        end = min(begin + config.batch_size, total)
        owners = torch.arange(begin, end, device=device) // config.trajectories_per_point
        points, owners = absorb(starting_points.detach()[owners].clone(), owners)
        for step in range(config.max_steps):
            if not len(points):
                break
            points = langevin_step(potential, points, beta, config.time_step, generator)
            points, owners = absorb(points, owners)
            if config.log_every and (step + 1) % config.log_every == 0:
                print(
                    f"Trajectories {begin + 1}-{end}: step {step + 1}, "
                    f"{len(points)} still active", flush=True,
                )
        unresolved.scatter_add_(0, owners, torch.ones_like(owners))
        if config.log_every:
            print(
                f"Processed {end}/{total} trajectories; "
                f"{int(unresolved.sum())} unresolved", flush=True,
            )

    return CommittorAnalysisResult(
        hits_a.cpu().numpy(), hits_b.cpu().numpy(), unresolved.cpu().numpy()
    )
