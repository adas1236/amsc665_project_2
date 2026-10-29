import torch


class SolutionModelNN(torch.nn.Module):
    """Scalar solution with inputs (t, x) and exact evenness in x."""

    def __init__(self, in_size=2, hidden_size=100, out_size=1, hidden_layers=6):
        super().__init__()
        layers = [torch.nn.Linear(in_size, hidden_size), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(hidden_size, out_size))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, tx):
        t, x = tx[:, 0:1], tx[:, 1:2]
        return self.layers(torch.cat((t, x.square()), dim=1))
