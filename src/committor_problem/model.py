from collections.abc import Callable

import torch


class CommittorModelNN(torch.nn.Module):
    def __init__(self, in_size: int, hidden_size: int = 25, out_size: int = 1, *, active_dimensions=None):
        super().__init__()
        if active_dimensions is not None:
            active_dimensions = tuple(active_dimensions)
        self.in_size, self.active_dimensions = in_size, active_dimensions
        self.linear0 = torch.nn.Linear(in_size if active_dimensions is None else len(active_dimensions), hidden_size)
        self.norm0 = torch.nn.LayerNorm(hidden_size)
        self.activation0 = torch.nn.Tanh()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.activation1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.activation2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(hidden_size, out_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_dimensions is not None:
            x = x[:, self.active_dimensions]
        q = self.activation0(self.norm0(self.linear0(x)))
        q = self.activation1(self.norm1(self.linear1(q)))
        q = self.activation2(self.norm2(self.linear2(q)))
        return self.sigmoid(self.linear3(q))


class BoundaryConditionModel(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        chi_a: Callable[[torch.Tensor], torch.Tensor],
        chi_b: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.network = network
        self.chi_a = chi_a
        self.chi_b = chi_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.network(x)
        chia = self.chi_a(x).reshape(-1, 1)
        chib = self.chi_b(x).reshape(-1, 1)
        return (1 - chia) * ((1 - chib) * q + chib)
