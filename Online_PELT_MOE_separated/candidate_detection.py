import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal

def reconstruction_loss(X_t, recon_experts, weights):
    """
    Calculate reconstruction error.
    """
    X_t = X_t.float()
    window_size = X_t.shape[1]
    with torch.no_grad():
        first_time_step_data = X_t[:, -1:, :]
        first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
        first_time_step_data = first_time_step_data.view(X_t.shape[0], -1)
        weighted_reconstruction = torch.zeros_like(first_time_step_data)
        num_experts_per_model = len(recon_experts[0]) if recon_experts else 0
        weight_idx = 0
        for model_experts in recon_experts:   # model_experts is [(Encoder, Decoder), ...]
            for encoder, decoder in model_experts:  # Directly unpack tuple
                if len(weights) > 3 :
                    print("weight_idx:",weights[weight_idx])
                    print("encoder:",encoder)
                    print("decoder:", decoder)
                # 计算重构
                latent = encoder(first_time_step_data)
                expert_reconstruction = decoder(latent)
                weighted_reconstruction += weights[weight_idx] * expert_reconstruction
                weight_idx += 1
        total_weight = sum(weights)
        weighted_reconstruction /= total_weight
        reconstruction_error = torch.nn.functional.mse_loss(
            weighted_reconstruction,
            first_time_step_data,
            reduction='none'
        ).mean(dim=1) / window_size
    return reconstruction_error.detach().cpu().numpy()

# def likelihood(X_t, feature_experts, weights, mean_list, cov_list):
#     """
#     Calculate likelihood in feature space.
#     """
#     X_t = X_t.float()
#     n = X_t.shape[0]
#     window_size = X_t.shape[1]
#     num_experts_per_model = int(len(weights)/len(feature_experts)) if feature_experts else 0
#     first_time_step_data = X_t[:, -1:, :]
#     first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
#     first_time_step_data = first_time_step_data.view(X_t.shape[0], -1)
#
#     likelihood_values = np.zeros(n)
#     weight_idx = 0
#
#     for i, model_experts in enumerate(feature_experts):
#         print("model_experts:",model_experts)
#         model_weights = weights[weight_idx:weight_idx + num_experts_per_model]
#         weight_idx += num_experts_per_model
#         model_means = mean_list[i]
#         model_covs = cov_list[i]
#
#         for weight, mean, cov in zip(model_weights, model_means, model_covs):
#             encoder = model_experts
#             feature_xt = encoder(first_time_step_data).detach().numpy()
#             cov += 1e-3 * np.eye(cov.shape[0])
#
#             for j in range(n):
#                 likelihood_values[j] += weight * multivariate_normal.pdf(
#                     feature_xt[j, :],
#                     mean=mean,
#                     cov=cov
#                 )
#
#     return likelihood_values
#
# def cosine_similarity(X_t, weights, train_mean, feature_experts):
#     """
#     Calculate cosine similarity.
#     """
#     X_t = X_t.float()
#     n = X_t.shape[0]
#     window_size = X_t.shape[1]
#     first_time_step_data = X_t[:, -1:, :]
#     first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
#     first_time_step_data = first_time_step_data.view(X_t.shape[0], -1)
#     first_time_step_data = first_time_step_data[:,:train_mean[0].shape[0]]
#     cos_sim = np.zeros(n)
#     num_experts_per_model = len(feature_experts[0])
#     for i in range(len(weights)):
#         for j in range(n):
#             cos_sim[j] += weights[i] * np.dot(first_time_step_data[j,:], train_mean[int(i/num_experts_per_model)][0]) / (np.linalg.norm(first_time_step_data[j,:]) * np.linalg.norm(train_mean[int(i/num_experts_per_model)]))
#     return cos_sim

def score(X_t, recon_experts, feature_experts, mean_list, cov_list, train_mean, weights, lambda1, lambda2, lambda3):
    """
    Calculate total detection score for new samples.
    """
    recon_loss = reconstruction_loss(X_t, recon_experts, weights)
    # likelihood_value = likelihood(X_t, feature_experts, weights, mean_list, cov_list)
    # cos_sim = cosine_similarity(X_t, weights, train_mean, feature_experts)
    score = lambda1 * recon_loss
    return score, recon_loss

def compute_scores_for_3d_input(X_3d, recon_experts, feature_experts, mean_list, cov_list, train_mean, weights, lambda1, lambda2, lambda3):
    """
    Calculate scores for each sample in 3D tensor input, returns array of shape (n,).
    """
    # n = X_3d.shape[0]
    # window_size = X_3d.shape[1]
    # scores = np.zeros(n)
    # recon_losses = np.zeros(n)
    # likelihood_values = np.zeros(n)
    # cos_sim = np.zeros(n)
    # for i in range(n):
    #     X_t = X_3d[i:i+1, -1:, :]
    #     X_t = X_t.repeat(1, window_size, 1)
    #     X_t = X_t.view(1, -1)
    #     scores[i],recon_losses[i],likelihood_values[i],cos_sim[i] = score(X_t, recon_experts, feature_experts, mean_list, cov_list, train_mean, weights, lambda1, lambda2, lambda3)
    scores,recon_losses = score(X_3d, recon_experts, feature_experts, mean_list, cov_list, train_mean, weights, lambda1, lambda2, lambda3)
    return scores, recon_losses

