import math

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(sizes, final_activation=False):
    layers = []
    for input_size, output_size in zip(sizes[:-1], sizes[1:]):
        layers.extend([nn.Linear(input_size, output_size), nn.Tanh()])
    return nn.Sequential(*(layers if final_activation else layers[:-1]))


def _output(raw, queries, x_bounds, kind):
    if kind == "sigmoid":
        return torch.sigmoid(raw)
    coordinate = (queries - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    return coordinate + 4 * coordinate * (1 - coordinate) * raw


class DeepONet(nn.Module):
    input_kind = "branch"

    def __init__(self, branch_size, latent_size=40, branch_width=80, trunk_width=40, output="hard_bc", x_bounds=(-1.0, 1.0)):
        super().__init__()
        self.branch = _mlp([branch_size, branch_width, branch_width, latent_size])
        self.trunk = _mlp([1, trunk_width, trunk_width, latent_size], final_activation=True)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.output, self.x_bounds = output, tuple(x_bounds)

    def forward(self, inputs, queries):
        branch = self.branch(inputs)
        normalized = 2 * (queries - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) - 1
        trunk = self.trunk(normalized.unsqueeze(-1))
        if queries.ndim == 1:
            raw = branch @ trunk.T + self.b0
        elif queries.ndim == 2 and queries.shape[0] == len(inputs):
            raw = torch.einsum("bp,bkp->bk", branch, trunk) + self.b0
        return _output(raw, queries, self.x_bounds, self.output)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.out_channels, self.modes = out_channels, modes
        scale = 1 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, 2))

    def forward(self, values):
        spectrum = torch.fft.rfft(values, dim=-1)
        count = min(self.modes, spectrum.shape[-1])
        weights = torch.view_as_complex(self.weight[:, :, :count].contiguous())
        transformed = spectrum.new_zeros((len(values), self.out_channels, spectrum.shape[-1]))
        transformed[:, :, :count] = torch.einsum("bim,iom->bom", spectrum[:, :, :count], weights)
        return torch.fft.irfft(transformed, n=values.shape[-1], dim=-1)


def _interpolate_uniform(values, queries, x_bounds):
    if queries.ndim == 1:
        queries = queries.unsqueeze(0).expand(len(values), -1)
    positions = (queries - x_bounds[0]) * (values.shape[1] - 1) / (x_bounds[1] - x_bounds[0])
    positions = positions.clamp(0, values.shape[1] - 1)
    left = positions.floor().long().clamp(max=values.shape[1] - 2)
    weight = positions - left
    return (1 - weight) * values.gather(1, left) + weight * values.gather(1, left + 1)


class FNO1D(nn.Module):
    input_kind = "force"
    requires_fft = True

    def __init__(self, width=32, modes=16, n_layers=4, padding_fraction=0.125, output="hard_bc", x_bounds=(-1.0, 1.0)):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))
        self.padding_fraction, self.output, self.x_bounds = padding_fraction, output, tuple(x_bounds)

    def forward(self, forces, queries):
        grid = torch.linspace(*self.x_bounds, forces.shape[1], dtype=forces.dtype, device=forces.device)
        coordinates = grid.unsqueeze(0).expand(len(forces), -1)
        features = torch.stack((forces, coordinates), dim=-1)
        values = self.lift(features).transpose(1, 2)
        padding = math.ceil(forces.shape[1] * self.padding_fraction)
        if padding:
            values = F.pad(values, (0, padding))
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            values = spectral(values) + local(values)
            if index + 1 < len(self.spectral):
                values = F.gelu(values)
        if padding:
            values = values[:, :, :-padding]
        raw = self.project(values.transpose(1, 2)).squeeze(-1)
        predicted_grid = _output(raw, coordinates, self.x_bounds, self.output)
        return _interpolate_uniform(predicted_grid, queries, self.x_bounds)
