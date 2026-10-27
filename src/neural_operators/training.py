from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from project_runtime import prepare_model

from .evaluation import operator_metrics, predict_operator


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 600
    batch_size: int = 64
    learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-5
    validation_every: int = 10
    log_every: int = 50
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class TrainingResult:
    history: dict[str, np.ndarray]
    best_epoch: int


def train_operator(model, training, validation, config=None):
    if config is None:
        config = TrainingConfig()
    parameter = prepare_model(model, config.device)
    inputs = training.inputs_for(model.input_kind)
    generator = torch.Generator().manual_seed(config.seed)
    n_batches = (len(inputs) + config.batch_size - 1) // config.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * n_batches, eta_min=config.minimum_learning_rate,
    )
    history = {
        "train_mse": np.zeros(config.epochs),
        "validation_mse": np.full(config.epochs, np.nan),
        "learning_rate": np.zeros(config.epochs),
    }
    best_loss, best_epoch, best_state = np.inf, -1, None
    validation_loss = np.nan
    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(inputs), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            queries = training.queries[indices] if training.queries.ndim == 2 else training.queries
            batch = inputs[indices].to(device=parameter.device, dtype=parameter.dtype)
            queries = queries.to(device=parameter.device, dtype=parameter.dtype)
            targets = training.targets[indices].to(device=parameter.device, dtype=parameter.dtype)
            optimizer.zero_grad()
            loss = (model(batch, queries) - targets).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            history["train_mse"][epoch] += loss.item() * len(indices) / len(inputs)
        history["learning_rate"][epoch] = scheduler.get_last_lr()[0]
        if epoch % config.validation_every == 0 or epoch + 1 == config.epochs:
            predicted = predict_operator(model, validation, config.batch_size)
            validation_loss = operator_metrics(predicted, validation.targets.cpu().numpy())["MSE"]
            history["validation_mse"][epoch] = validation_loss
            if validation_loss < best_loss:
                best_loss, best_epoch, best_state = validation_loss, epoch, deepcopy(model.state_dict())
        if config.log_every and (epoch % config.log_every == 0 or epoch + 1 == config.epochs):
            print(
                f"Epoch {epoch}: train RMSE={np.sqrt(history['train_mse'][epoch]):.3e}, "
                f"validation RMSE={np.sqrt(validation_loss):.3e}", flush=True,
            )
    model.load_state_dict(best_state)
    return TrainingResult(history, best_epoch)


def compare_operators(models, splits, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, report = {}, {}
    for name, model in models.items():
        print(f"Training {name}", flush=True)
        start = time.perf_counter()
        fitted = train_operator(model, splits.train, splits.validation, config)
        elapsed = time.perf_counter() - start
        metrics, predictions = {}, {}
        for split_name in ("train", "validation", "test"):
            data = getattr(splits, split_name)
            predictions[split_name] = predict_operator(model, data)
            metrics[split_name] = operator_metrics(predictions[split_name], data.targets.cpu().numpy())
        record = {
            "metrics": metrics, "best_epoch": fitted.best_epoch,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": elapsed,
        }
        report[name] = record
        results[name] = {**record, "history": fitted.history, "predictions": predictions["test"]}
        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        np.savez(output_dir / f"{name}_history.npz", **fitted.history)
        np.savez_compressed(
            output_dir / f"{name}_test.npz", queries=splits.test.queries.cpu().numpy(),
            reference=splits.test.targets.cpu().numpy(), predicted=predictions["test"],
        )
        print(f"{name}: train RMSE={metrics['train']['RMSE']:.3e}, test RMSE={metrics['test']['RMSE']:.3e}", flush=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return results
