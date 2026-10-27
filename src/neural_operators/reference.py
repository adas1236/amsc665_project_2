import numpy as np
from scipy.fft import dct
from scipy.interpolate import CubicSpline


class ChebyshevCommittor:
    def __init__(self, degree=64, beta=1.0, x_bounds=(-1.0, 1.0)):
        left, right = x_bounds
        self.degree, self.beta, self.x_bounds = degree, beta, tuple(x_bounds)
        z = np.cos(np.pi * np.arange(degree + 1) / degree)
        c = np.r_[2.0, np.ones(degree - 1), 2.0] * (-1.0)**np.arange(degree + 1)
        difference = z[:, None] - z[None, :]
        derivative = (c[:, None] / c[None, :]) / (difference + np.eye(degree + 1))
        derivative -= np.diag(derivative.sum(axis=1))
        self.nodes = (left + right) / 2 + (right - left) * z / 2
        self.derivative = derivative * 2 / (right - left)
        self.second_derivative = self.derivative @ self.derivative

    def solve(self, force_at_nodes, queries):
        forces = np.asarray(force_at_nodes, dtype=float)
        single = forces.ndim == 1
        forces = np.atleast_2d(forces)
        queries = np.asarray(queries, dtype=float)
        left, right = self.x_bounds
        matrix = self.second_derivative[None] - self.beta * forces[:, :, None] * self.derivative[None]
        matrix[:, 0, :] = 0
        matrix[:, -1, :] = 0
        matrix[:, 0, 0] = 1
        matrix[:, -1, -1] = 1
        rhs = np.zeros((len(forces), len(self.nodes), 1))
        rhs[:, 0, 0] = 1
        nodal_values = np.linalg.solve(matrix, rhs)[..., 0]
        coefficients = dct(nodal_values, type=1, axis=1) / self.degree
        coefficients[:, [0, -1]] *= 0.5
        normalized = 2 * (queries - left) / (right - left) - 1
        values = np.polynomial.chebyshev.chebval(
            normalized, coefficients.T[:, :, None], tensor=False
        )
        values = np.where(queries == left, 0.0, np.where(queries == right, 1.0, values))
        return values[0] if single else values

    def from_samples(self, forces, sensor_grid, queries, batch_size=128):
        forces = np.asarray(forces, dtype=float)
        single = forces.ndim == 1
        forces = np.atleast_2d(forces)
        grid = np.asarray(sensor_grid, dtype=float)
        queries = np.asarray(queries, dtype=float)
        values = []
        for start in range(0, len(forces), batch_size):
            end = start + batch_size
            interpolated = CubicSpline(grid, forces[start:end], axis=1)(self.nodes)
            y = queries[start:end] if queries.ndim == 2 else queries
            values.append(self.solve(interpolated, y))
        result = np.concatenate(values)
        return result[0] if single else result
