import numpy as np
import torch
import torch.nn as nn

class GradKPCA(nn.Module):
    def __init__(self, input_size, n_components, latent_space_size, batch_size, kernel_type='rbf'):
        super(GradKPCA, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_components = n_components
        self.W = torch.nn.Parameter(0.05 * torch.randn(batch_size, self.n_components), requires_grad=True)
        self.gamma = 10
        self.save_x = None
        self.is_3d = False
        self.feed_forward = nn.Sequential(
            nn.Linear(n_components, 4 * latent_space_size),
            nn.ReLU(),
            nn.Linear(4 * latent_space_size, latent_space_size),
        )
        self.kernel_matrix = None
        # self.norm = nn.LayerNorm(latent_space_size)
        self.shortcut = nn.Linear(n_components, latent_space_size)

        self.kernel_functions = {
            'linear': self.linear_kernel,
            'rbf': self.rbf_kernel
        }

        if kernel_type not in self.kernel_functions:
            raise ValueError(f'kernel_type must be one of {list(self.kernel_functions.keys())}')
        self.kernel_type = kernel_type
        self.kernel = self.kernel_functions[self.kernel_type]

    def rbf_kernel(self, x, gamma=None):
        if gamma is None:
            gamma = self.gamma
        pairwise_sq_dists = torch.cdist(x, self.save_x, p=2) ** 2
        kernel_matrix = torch.exp(-pairwise_sq_dists / (gamma * x.shape[1]))
        return kernel_matrix

    def linear_kernel(self, x, gamma=None):
        return x @ self.save_x.T / x.shape[1]

    def forward(self, x):
        if self.save_x is None:
            self.save_x = x
        if self.W is None:
            self.W = torch.nn.Parameter(0.5 * torch.randn(x.shape[0], self.n_components).cuda(), requires_grad=True)
        # print("x:",x.shape)
        kernel_matrix = self.center_kernel_matrix(self.kernel(x, gamma=self.gamma))
        self.kernel_matrix = kernel_matrix
        score = kernel_matrix @ self.W
        z = self.feed_forward(score) + self.shortcut(score)
        return z

    def self_loss(self, x, labels=None):
        if labels is None:
            pad_size = self.save_x.shape[0] - x.shape[0]
            if pad_size > 0:
                padding = torch.zeros(pad_size, x.shape[1], dtype=x.dtype, device=x.device)
                pad_x = torch.cat([x, padding], dim=0)
            else:
                pad_x = x
            K = self.center_kernel_matrix(self.kernel(pad_x, gamma=self.gamma))
            projection_loss = (torch.trace(K) - torch.trace(self.W.T @ K @ self.W)) / pad_x.shape[0]
            W_heb = self.W.clone().detach()
            U = torch.tril(W_heb.T @ K @ W_heb)
            hebbian_loss = torch.trace(self.W @ U.T @ self.W.T) / pad_x.shape[0]
            loss = projection_loss + hebbian_loss
            loss.data = projection_loss.data
        else:
            indices = torch.nonzero(labels.any(dim=1)).squeeze()
            selected_rows = x[indices]
            pad_size = self.save_x.shape[0] - selected_rows.shape[0]
            if pad_size > 0:
                padding = torch.zeros(pad_size, selected_rows.shape[1], dtype=selected_rows.dtype, device=selected_rows.device)
                pad_x = torch.cat([selected_rows, padding], dim=0)
            else:
                pad_x = selected_rows
            K = self.center_kernel_matrix(self.kernel(pad_x, gamma=self.gamma))
            projection_loss = (torch.trace(K) - torch.trace(self.W.T @ K @ self.W)) / pad_x.shape[0]
            W_heb = self.W.clone().detach()
            U = torch.tril(W_heb.T @ K @ W_heb)
            hebbian_loss = torch.trace(self.W @ U.T @ self.W.T) / pad_x.shape[0]
            loss = projection_loss + hebbian_loss
            loss.data = projection_loss.data
        return loss

    def residual(self, x):
        x = x.reshape(x.shape[0], -1)
        pad_size = self.save_x.shape[0] - x.shape[0]
        if pad_size > 0:
            padding = torch.zeros(pad_size, x.shape[1], dtype=x.dtype, device=x.device)
            pad_x = torch.cat([x, padding], dim=0)
        else:
            pad_x = x
        K = self.center_kernel_matrix(self.kernel(pad_x, gamma=self.gamma))
        return torch.diag(K - K @ self.W @ self.W.T)[:x.shape[0]]

    @staticmethod
    def center_kernel_matrix(K):
        n_left = K.size(0)
        n_right = K.size(1)
        left_one_n = torch.ones((n_left, n_left), device=K.device) / n_left
        right_one_n = torch.ones((n_right, n_right), device=K.device) / n_right
        # print("K:",K.shape)
        # print("left_one_n:", left_one_n.shape)
        K_centered = K - left_one_n @ K - K @ right_one_n + left_one_n @ K @ right_one_n
        return K_centered
