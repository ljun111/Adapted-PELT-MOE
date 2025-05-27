import torch
import torch.nn as nn
import numpy as np
from Online_PELT_MOE_separated.online_detection_separated import segment_online
from Encoder.Encoder import Encoder_base,Decoder_base
from Encoder.LSTM import EncoderLSTM
from Online_PELT_MOE_separated.Encoder_train import train_encoder_decoder, reshape_to_3d

def train_and_segment_online(
    model_name: str,
    data_train: torch.Tensor,
    data: torch.Tensor,
    latent_space_size: int,
    hidden_dim: int,
    window_size: int,
    min_seg_len: int,
    K: int,
    beta: float,
    gamma: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    tau: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    n_components: int
):
    """
    Combined function for training, change point detection, and online segmentation.

    Args:
        model_name (str): Name of the model.
        data_train (torch.Tensor): Training data with shape (n_train, L, P).
        data (torch.Tensor): Online segmentation data with shape (n, L, P).
        latent_space_size (int): Dimension of latent space.
        hidden_dim (int): Dimension of hidden layer.
        window_size (int): Size of sliding window.
        min_seg_len (int): Minimum segment length.
        K (int): Number of experts.
        beta (float): Hyperparameter for change point detection.
        gamma (float): Hyperparameter for online segmentation.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        tau (float): Hyperparameter for online segmentation.
        lambda1 (float): Regularization parameter 1.
        lambda2 (float): Regularization parameter 2.
        lambda3 (float): Regularization parameter 3.
        n_components (int): Number of components.
    """

    results = train_encoder_decoder(
        model_name=model_name,
        data_train=data_train,
        epochs=epochs,
        learning_rate=learning_rate,
        latent_space_size=latent_space_size,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        window_size=window_size,
        n_components=n_components
    )

    feature_experts = [[] for _ in range(len(model_name))]
    recon_experts = [[] for _ in range(len(model_name))]
    train_data_means = [[] for _ in range(len(model_name))]
    feature_means = [[] for _ in range(len(model_name))]
    feature_covariances = [[] for _ in range(len(model_name))]
    updated_covariances = []

    for i, model in enumerate(model_name):
        encoder = results[model]['encoder']
        decoder = results[model]['decoder']

        feature_experts[i].append(encoder)
        recon_experts[i].append((encoder, decoder))

        with torch.no_grad():
            data_train_tensor = reshape_to_3d(data_train, window_size)
            train_data_means[i] = torch.mean(data_train_tensor, dim=[0, 1]).detach().cpu().numpy()

            if model == 'GradTensorCP':
                data_train_tensor = reshape_to_3d(data_train,window_size)
                features = encoder(data_train_tensor).detach().numpy()
            else:
                features = encoder(torch.FloatTensor(data_train)).detach().numpy()

            feature_means[i] = np.mean(features, axis=0)
            feature_covariances[i] = np.cov(features, rowvar=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.from_numpy(data).to(device)
    changepoints_model = [[] for _ in range(len(model_name))]
    for i, model in enumerate(model_name):
        model_temp = [model]
        recon_experts_temp = [recon_experts[i]]
        feature_experts_temp = [feature_experts[i]]

        changepoints_online, y_pred = segment_online(
            model_name=model_temp,
            data=data,
            min_seg_len=min_seg_len,
            K=K,
            beta=beta,
            gamma=gamma,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            latent_space_size=latent_space_size,
            hidden_dim=hidden_dim,
            window_size=window_size,
            recon_experts=recon_experts_temp,
            feature_experts=feature_experts_temp,
            feature_means=feature_means,
            feature_covariances=updated_covariances,
            train_data_means=train_data_means,
            tau=tau,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            n_components=n_components
        )
        changepoints_model[i].append(changepoints_online)

    return changepoints_model
