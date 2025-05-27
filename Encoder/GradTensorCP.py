import torch
import torch.nn as nn

class GradTensorCP(nn.Module):
    def __init__(self, seq_len, input_size, n_ranks, latent_space_size):
        super(GradTensorCP, self).__init__()
        self.seq_len = seq_len
        self.a = torch.nn.Parameter(0.1 * torch.randn(seq_len, n_ranks))
        self.b = torch.nn.Parameter(0.1 * torch.randn(input_size, n_ranks))
        self.is_ml = False
        self.is_3d = True
        self.shortcut = nn.Linear(n_ranks, latent_space_size)
        self.n_ranks = n_ranks
        self.feed_forward = nn.Sequential(
            nn.Linear(n_ranks, 4 * latent_space_size),
            nn.ReLU(),
            nn.Linear(4 * latent_space_size, latent_space_size),
        )
        self.norm = nn.LayerNorm(latent_space_size)

    def _auto_reshape(self, x):
        if x.ndim == 2:
            batch_size = x.shape[0]
            features = x.shape[1]
            if features % self.seq_len != 0:
                raise RuntimeError(f'Input data error!!!')
            return x.view(batch_size, self.seq_len, features // self.seq_len)
        elif x.ndim == 3:
            return x
        else:
            raise RuntimeError(f'Input should be 2D or 3D，but get {x.ndim}D')

    def forward(self, x):
        x = self._auto_reshape(x)
        rk1_vectors = torch.einsum('ik,jk->ijk', self.project(self.a), self.project(self.b)).reshape(-1, self.n_ranks)
        x_vector = x.reshape(x.shape[0], -1)
        Lambda = x_vector @ torch.pinverse(rk1_vectors.T)
        z = self.shortcut(Lambda)
        return z

    def self_loss(self, x):
        x_vector = x.reshape(x.shape[0], -1)
        rk1_vectors = torch.einsum('ik,jk->ijk', self.project(self.a), self.project(self.b)).reshape(-1, self.n_ranks)
        Lambda = x_vector @ torch.pinverse(rk1_vectors.T)
        return torch.mean((x_vector - Lambda @ rk1_vectors.T) ** 2)

    def residual(self, x):
        B, L, C = x.shape
        x_vector = x.reshape(x.shape[0], -1)
        rk1_vectors = torch.einsum('ik,jk->ijk', self.project(self.a), self.project(self.b)).reshape(-1, self.n_ranks)
        Lambda = x_vector @ torch.pinverse(rk1_vectors.T)
        return torch.mean(((x_vector - Lambda @ rk1_vectors.T) ** 2).reshape(B, L, C), dim=2)

    @staticmethod
    def project(W):
        return W / torch.norm(W, p=2, dim=0, keepdim=True)
