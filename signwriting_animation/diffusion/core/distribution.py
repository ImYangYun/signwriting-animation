import torch
from torch import nn

class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.std = torch.exp(0.5 * logvar)
        self.distribution = torch.distributions.Normal(self.mean, self.std)

    def sample(self):
        return self.distribution.rsample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def nll(self, value):
        return -self.log_prob(value).mean()

class DistributionPredictionModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_size, 1)
        self.fc_var = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return DiagonalGaussianDistribution(mu, log_var)