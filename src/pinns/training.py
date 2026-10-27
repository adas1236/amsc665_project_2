from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from project_runtime import prepare_model, synchronize_device

from .sampling import TrainingPoints


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 2000
    batch_size: int = 312
    learning_rate: float = 1e-3
    milestones: tuple[int, ...] = (150, 300, 450)
    gamma: float = 0.5
    initial_weight: float = 10.0
    boundary_weight: float = 1.0
    pde_weight: float = 1.0
    seed: int = 0
    log_every: int = 100
    device: str = "auto"


def _gradient(values, coordinates):
    if not values.requires_grad:
        return torch.zeros_like(coordinates)
    result = torch.autograd.grad(
        values, coordinates, grad_outputs=torch.ones_like(values),
        create_graph=True, allow_unused=True,
    )[0]
    return torch.zeros_like(coordinates) if result is None else result


def solution_derivatives(model, tx):
    """Return u, u_t, u_x, and u_xx for a scalar solution with inputs (t, x)."""
    coordinates = tx.detach().requires_grad_(True)
    u = model(coordinates)
    first = _gradient(u, coordinates)
    u_x = first[:, 1:2]
    u_xx = _gradient(u_x, coordinates)[:, 1:2]
    return u, first[:, 0:1], u_x, u_xx


def periodic_boundary_loss(model, left, right):
    left = left.detach().requires_grad_(True)
    right = right.detach().requires_grad_(True)
    u_left, u_right = model(left), model(right)
    ux_left = _gradient(u_left, left)[:, 1:2]
    ux_right = _gradient(u_right, right)[:, 1:2]
    return (u_left - u_right).square().mean() + (ux_left - ux_right).square().mean()


def loss_components(model, interior, points: TrainingPoints, initial_condition, equation):
    residual = equation(*solution_derivatives(model, interior))
    initial = model(points.initial)
    target = initial_condition(points.initial[:, 1:2]).detach().reshape_as(initial)
    return {
        "pde": residual.square().mean(),
        "initial": (initial - target).square().mean(),
        "boundary": periodic_boundary_loss(model, points.boundary_left, points.boundary_right),
    }


def train(model, points: TrainingPoints, initial_condition, equation, config=None):
    """Fit the PINN and return losses plus per-epoch and cumulative loop timings.

    Timings include batching, all losses and derivatives, backward passes,
    optimizer updates, and statistics. Setup, reference solves, prediction,
    and saving are excluded. There is no validation pass in this trainer.
    """
    if config is None:
        config = TrainingConfig()

    parameter = prepare_model(model, config.device)
    points = points.to(parameter.device, parameter.dtype)

    generator = torch.Generator().manual_seed(config.seed)
    n_batches = (len(points.interior) + config.batch_size - 1) // config.batch_size
    if config.log_every:
        hardware = f" ({torch.cuda.get_device_name(parameter.device)})" if parameter.is_cuda else ""
        print(
            f"PINN training: model={parameter.device}{hardware}, interior={points.interior.device}, "
            f"initial={points.initial.device}, boundaries={points.boundary_left.device}/{points.boundary_right.device}, "
            f"dtype={parameter.dtype}; {n_batches} updates/epoch, batch size {config.batch_size}",
            flush=True,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=config.gamma
    )
    loss_names = ("pde", "initial", "boundary", "total")
    history = {name: np.zeros(config.epochs) for name in (*loss_names, "epoch_seconds", "elapsed_seconds")}
    model.train()
    synchronize_device(parameter.device)
    training_start = previous_log_time = perf_counter()
    previous_log_epochs = 0
    for epoch in range(config.epochs):
        epoch_start = perf_counter()
        epoch_losses = parameter.new_zeros(len(loss_names))
        permutation = torch.randperm(len(points.interior), generator=generator).to(parameter.device)
        for start in range(0, len(points.interior), config.batch_size):
            interior = points.interior[permutation[start:start + config.batch_size]]
            optimizer.zero_grad()
            losses = loss_components(model, interior, points, initial_condition, equation)
            total = (
                config.pde_weight * losses["pde"]
                + config.initial_weight * losses["initial"]
                + config.boundary_weight * losses["boundary"]
            )
            total.backward()
            if config.log_every and epoch == 0 and start == 0:
                gradient_devices = sorted({str(p.grad.device) for p in model.parameters() if p.grad is not None})
                memory = (
                    f", CUDA allocated={torch.cuda.memory_allocated(parameter.device) / 2**20:.1f} MiB"
                    if parameter.is_cuda else ""
                )
                print(
                    f"First backward pass: batch={interior.device}, loss={total.device}, "
                    f"gradients={','.join(gradient_devices)}{memory}", flush=True,
                )
            optimizer.step()
            losses["total"] = total
            weight = len(interior) / len(points.interior)
            epoch_losses += weight * torch.stack([losses[name].detach() for name in loss_names])
        values = epoch_losses.cpu().numpy()

        for name, value in zip(loss_names, values):
            history[name][epoch] = value
        scheduler.step()
        synchronize_device(parameter.device)
        epoch_end = perf_counter()
        history["epoch_seconds"][epoch] = epoch_end - epoch_start
        history["elapsed_seconds"][epoch] = epoch_end - training_start
        if config.log_every and (epoch % config.log_every == 0 or epoch + 1 == config.epochs):
            completed_epochs = epoch + 1
            interval_epochs = completed_epochs - previous_log_epochs
            interval_seconds = epoch_end - previous_log_time
            epochs_per_minute = 60 * interval_epochs / interval_seconds
            print(
                f"Epoch {epoch}: PDE={history['pde'][epoch]:.4e}, "
                f"IC={history['initial'][epoch]:.4e}, BC={history['boundary'][epoch]:.4e}, "
                f"device={parameter.device}, epoch_train={history['epoch_seconds'][epoch]:.2f}s, "
                f"train_total={history['elapsed_seconds'][epoch]:.2f}s; "
                f"last {interval_epochs} epochs={interval_seconds:.2f}s ({epochs_per_minute:.1f} epochs/min)",
                flush=True,
            )
            previous_log_time, previous_log_epochs = epoch_end, completed_epochs
    return history
