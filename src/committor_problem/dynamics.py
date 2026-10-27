import math

import torch
from project_runtime import normal_noise


def langevin_step(potential, points, beta, time_step, generator, extra_gradient=None):
    with torch.enable_grad():
        x = points.detach().requires_grad_(True)
        gradient = torch.autograd.grad(potential(x).sum(), x)[0]
    with torch.no_grad():
        if extra_gradient is not None:
            gradient = gradient + extra_gradient
        noise = normal_noise(points, generator)
        result = points - time_step * gradient + math.sqrt(2 * time_step / beta) * noise
    return result
