import torch
import torch.nn as nn

class GradPCA(nn.Module):
    def __init__(self, input_size, n_components, latent_space_size):
        super(GradPCA, self).__init__()
        self.W = torch.nn.Parameter(0.1 * torch.randn(input_size, n_components))
        self.is_ml = False
        self.is_3d = False
        self.sum_B = None
        self.cov_score = None
        self.shortcut = nn.Linear(n_components, latent_space_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_components, 4 * latent_space_size),
            nn.ReLU(),
            nn.Linear(4 * latent_space_size, latent_space_size),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError('x should be a 2d tensor.')
        W_projected, _ = torch.qr(self.W)
        pc_score = x @ W_projected
        # z = self.feed_forward(pc_score)
        # z = self.norm(z + self.shortcut(pc_score))
        z = self.shortcut(pc_score)
        # z = self.norm(z)
        return z

    def self_loss(self, x):
        covariance_matrix = x.T @ x / x.shape[0]
        W_projected, _ = torch.qr(self.W)
        return torch.trace(covariance_matrix) / x.shape[1] - torch.trace(W_projected.T @ covariance_matrix @ W_projected) / x.shape[1]

    @torch.no_grad()
    def update_statistic(self, x):
        B, L, C = x.shape
        fx = x.reshape(x.shape[0], -1)
        W_projected, _ = torch.qr(self.W)
        if self.cov_score is None:
            residuals2 = ((fx - fx @ W_projected @ W_projected.T) ** 2).mean(dim=1)
            self.cov_score = (residuals2 ** 2).sum() / B
            self.sum_B = B
        else:
            residuals2 = ((fx - fx @ W_projected @ W_projected.T) ** 2).mean(dim=1)
            # self.mean = (self.mean * self.sum_B * 0.9 + score.mean(dim=0) * B) / (self.sum_B * 0.9 + B)
            # center_score = score - score.mean(dim=0).unsqueeze(0)
            self.cov_score = (self.cov_score * self.sum_B + residuals2.T @ residuals2) / (
                        self.sum_B + B)
            self.sum_B = self.sum_B + B

    def residual(self, x):
        B, L, C = x.shape
        x = x.reshape(x.shape[0], -1)
        W_projected, _ = torch.qr(self.W)
        return ((x - x @ W_projected @ W_projected.T) ** 2).reshape(B, L, C).mean(dim=2) # / self.cov_score

    def self_residual(self, x):
        B, L, C = x.shape
        x = x.reshape(x.shape[0], -1)
        W_projected, _ = torch.qr(self.W)
        score = (x @ W_projected)
        center_score = score - self.mean.unsqueeze(0)
        anoma_score = torch.diag(center_score @ torch.linalg.inv(self.cov_score) @ center_score.T) / x.shape[1]
        return anoma_score.sum(), anoma_score.unsqueeze(1).expand(-1, L)
