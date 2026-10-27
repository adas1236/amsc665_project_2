import torch


class SolutionModelNN(torch.nn.Module):
    def __init__(self, in_size=2, hidden_size=100, out_size=1, hidden_layers=4):
        super().__init__()
        layers = [torch.nn.Linear(in_size, hidden_size), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(hidden_size, out_size))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, tx):
        return self.layers(tx)
