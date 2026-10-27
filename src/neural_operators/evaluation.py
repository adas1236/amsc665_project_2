import numpy as np
import torch


def predict_operator(model, data, batch_size=128):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    parameter = next(model.parameters())
    inputs = data.inputs_for(model.input_kind)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            end = start + batch_size
            queries = data.queries[start:end] if data.queries.ndim == 2 else data.queries
            predictions.append(model(
                inputs[start:end].to(device=parameter.device, dtype=parameter.dtype),
                queries.to(device=parameter.device, dtype=parameter.dtype),
            ).cpu().numpy())
    return np.concatenate(predictions)


def operator_metrics(predicted, reference):
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    mse = float(np.mean((predicted - reference)**2))
    return {"MSE": mse, "RMSE": float(np.sqrt(mse))}
