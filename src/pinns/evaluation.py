import numpy as np
import torch


def predict_on_grid(model, t, x, batch_size=8192):
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
    device = parameter.device if parameter is not None else torch.device("cpu")
    t_grid, x_grid = np.meshgrid(t, x)
    coordinates = np.column_stack((t_grid.ravel(), x_grid.ravel()))
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coordinates), batch_size):
            points = torch.as_tensor(coordinates[start:start + batch_size], dtype=dtype, device=device)
            predictions.append(model(points).cpu().numpy().reshape(-1))
    return np.concatenate(predictions).reshape(len(x), len(t))


def error_metrics(predicted, reference):
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    error = predicted - reference
    return {"MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error**2)))}
