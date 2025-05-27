import torch
import torch.nn as nn
import torch.nn.functional as F

class GradDict(nn.Module):
    def __init__(self, input_size, n_components, latent_space_size, lambda0=1e-2):
        super(GradDict, self).__init__()
        self.D = torch.nn.Parameter(0.1 * torch.randn(input_size, n_components))
        self.is_ml = False
        self.is_3d = False
        self.shortcut = nn.Linear(n_components, latent_space_size)
        self.norm = nn.LayerNorm(latent_space_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_components, 4 * latent_space_size),
            nn.ReLU(),
            nn.Linear(4 * latent_space_size, latent_space_size),
        )
        self.lambda0 = lambda0

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError('x should be a 2d tensor')
        proj_D = self.project(self.D)
        A = x @ torch.pinverse(proj_D.T)
        z = self.shortcut(A) # + self.feed_forward(A)
        return z

    def self_loss(self, x):
        proj_D = self.project(self.D)
        A = x @ torch.pinverse(proj_D.T)
        loss = F.mse_loss(x, A @ proj_D.T)
        # print(loss)
        return loss + self.lambda0 * torch.mean(torch.norm(proj_D, p=1, dim=0))

    @staticmethod
    def project(D):
        return D / torch.norm(D, p=2, dim=0, keepdim=True)