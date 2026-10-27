from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq


@dataclass(frozen=True)
class SpectralConfig:
    n_space: int = 2048
    n_steps: int = 100_000
    save_every: int = 100
    log_every: int = 10_000


@dataclass(frozen=True)
class SpectralSolution:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray


def solve_spectral(initial_condition, linear_symbol, nonlinear_term, x_bounds, t_bounds, config=None):
    """Fourier integrating-factor RK4 for a real periodic scalar PDE."""
    if config is None:
        config = SpectralConfig()
    xmin, xmax = x_bounds
    tmin, tmax = t_bounds

    x = np.linspace(xmin, xmax, config.n_space, endpoint=False)
    frequencies = 2 * np.pi * rfftfreq(config.n_space, d=(xmax - xmin) / config.n_space)
    dt = (tmax - tmin) / config.n_steps
    eigenvalues = np.asarray(linear_symbol(frequencies))

    half_step = np.exp(0.5 * dt * eigenvalues)
    full_step = half_step**2
    initial = np.asarray(initial_condition(x))

    initial = initial.astype(np.float64)
    state = rfft(initial)
    save_steps = np.unique(np.r_[0, np.arange(config.save_every, config.n_steps + 1, config.save_every), config.n_steps])
    times = tmin + save_steps * dt
    values = np.empty((config.n_space, len(times)))
    values[:, 0] = initial
    next_save = 1

    def nonlinear_fourier(fourier_values):
        return rfft(nonlinear_term(irfft(fourier_values, n=config.n_space)))

    for step in range(1, config.n_steps + 1):
        a = dt * nonlinear_fourier(state)
        b = dt * nonlinear_fourier(half_step * (state + 0.5 * a))
        c = dt * nonlinear_fourier(half_step * state + 0.5 * b)
        d = dt * nonlinear_fourier(full_step * state + half_step * c)
        state = full_step * state + (full_step * a + 2 * half_step * (b + c) + d) / 6
        if step == save_steps[next_save]:
            values[:, next_save] = irfft(state, n=config.n_space)
            next_save += 1
        if config.log_every and step % config.log_every == 0:
            print(f"Spectral steps: {step}/{config.n_steps}", flush=True)

    return SpectralSolution(x, times, values)
