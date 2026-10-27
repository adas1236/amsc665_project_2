from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np
import torch
from project_runtime import make_generator, move_tensor, prepare_model

from .sampling import refresh_gaussian_coordinates


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 400
    batch_size: int = 256
    learning_rate: float = 1e-3
    milestones: tuple[int, ...] = (100, 200, 300)
    gamma: float = 0.5
    log_every: int = 20
    seed: int = 0
    device: str = "auto"
    refresh_dimensions: tuple[int, ...] = ()
    refresh_standard_deviation: float | None = None


def dirichlet_loss(
    model: torch.nn.Module,
    points: torch.Tensor,
    potential: Callable[[torch.Tensor], torch.Tensor],
    beta: float | torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    points = points.detach().requires_grad_(True)
    q = model(points)
    dq = torch.autograd.grad(
        q, points, grad_outputs=torch.ones_like(q), create_graph=True
    )[0]
    if sample_weights is None:
        with torch.no_grad():
            values = potential(points).reshape(-1)
            weights = torch.exp(-torch.as_tensor(beta, device=points.device, dtype=points.dtype) * values)
    else:
        weights = sample_weights.detach()
    integrand = dq.pow(2).sum(dim=1) * weights
    return integrand.mean()


def train(
    model: torch.nn.Module,
    points: torch.Tensor,
    potential: Callable[[torch.Tensor], torch.Tensor],
    beta: float | torch.Tensor,
    config: TrainingConfig | None = None,
    *,
    sample_weights: torch.Tensor | None = None,
) -> np.ndarray:
    if config is None:
        config = TrainingConfig()
    parameter = prepare_model(model, config.device)
    points = move_tensor(points.detach(), parameter.device, dtype=parameter.dtype)
    weights = None
    if sample_weights is None:
        weight_device = "computed on the training device"
    else:
        weights = move_tensor(sample_weights.detach(), parameter.device, dtype=parameter.dtype)
        if (weights.shape != (len(points),) or not torch.isfinite(weights).all()
                or (weights < 0).any() or not (weights > 0).any()):
            raise ValueError("Sample weights must be finite, nonnegative, and have positive total mass.")
        weights = weights / weights.max()
        weights = weights / weights.mean()
        weight_device = str(weights.device)

    if config.log_every:
        hardware = f" ({torch.cuda.get_device_name(parameter.device)})" if parameter.is_cuda else ""
        print(
            f"Committor training: model={parameter.device}{hardware}, "
            f"points={points.device}, weights={weight_device}, dtype={parameter.dtype}; "
            f"{len(points):,} points, batch size {config.batch_size:,}", flush=True,
        )
        if config.refresh_dimensions:
            print(
                f"Independent Gaussian refresh: columns {config.refresh_dimensions}, "
                f"standard deviation={config.refresh_standard_deviation:.6g}", flush=True,
            )

    generator = torch.Generator().manual_seed(config.seed)
    refresh_generator = make_generator(points.device, config.seed + 1) if config.refresh_dimensions else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=config.gamma
    )
    loss_epoch_average = np.zeros(config.epochs)

    model.train()
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        loss_sum = points.new_zeros(())
        permutation = torch.randperm(len(points), generator=generator).to(points.device)
        for start in range(0, len(points), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            x = points[indices]
            if refresh_generator is not None:
                x = refresh_gaussian_coordinates(
                    x, config.refresh_dimensions, config.refresh_standard_deviation, refresh_generator,
                )
            optimizer.zero_grad()
            loss = dirichlet_loss(
                model, x, potential, beta, sample_weights=weights[indices] if weights is not None else None,
            )
            loss.backward()
            if config.log_every and epoch == 0 and start == 0:
                gradient_devices = sorted({str(p.grad.device) for p in model.parameters() if p.grad is not None})
                memory = (
                    f", CUDA allocated={torch.cuda.memory_allocated(parameter.device) / 2**20:.1f} MiB"
                    if parameter.is_cuda else ""
                )
                print(
                    f"First backward pass: batch={x.device}, loss={loss.device}, "
                    f"gradients={','.join(gradient_devices)}{memory}", flush=True,
                )
            optimizer.step()
            loss_sum += loss.detach() * len(x)

        loss_epoch_average[epoch] = (loss_sum / len(points)).item()
        scheduler.step()
        if config.log_every and epoch % config.log_every == 0:
            print(
                f"epoch = {epoch}, average loss = {loss_epoch_average[epoch]:.6e}, "
                f"lr = {scheduler.get_last_lr()[0]:.1e}, "
                f"device = {parameter.device}, elapsed = {time.perf_counter() - epoch_start:.2f}s",
                flush=True,
            )

    return loss_epoch_average


def predict(model: torch.nn.Module, points: torch.Tensor, batch_size=8192) -> np.ndarray:
    parameter = next(model.parameters())
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model(move_tensor(batch, parameter.device, dtype=parameter.dtype)).cpu().numpy().flatten()
            for batch in points.split(batch_size)
        ])
