import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Adagrad, SGD
import numpy as np
from Encoder.Encoder import Encoder_base, Encoder_base_2, Encoder_base_3, Encoder_base_4, Encoder_base_5, Decoder_base
from Encoder.LSTM import EncoderLSTM
from Encoder.GradKPCA import GradKPCA
from Encoder.GradPCA import GradPCA
from Encoder.GradTensorCP import GradTensorCP
from Encoder.GradDict import GradDict
from torch.utils.data import DataLoader, TensorDataset

def train_encoder_decoder(model_name, data_train, epochs, learning_rate, latent_space_size, hidden_dim, batch_size, window_size, n_components):
    """Train encoder and decoder.

    Args:
        data_train (array): train data.
        epochs (int): train epoch.
        learning_rate (float): learning rate

    Returns:
        float: quantile of losses for each sample in the last epoch.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2035)

    # Prepare data
    data_train_tensor = reshape_to_3d(data_train, window_size)
    data_train_tensor = data_train_tensor.view(data_train.shape[0], -1)
    dataset_merge = TensorDataset(data_train_tensor)
    data_loader = DataLoader(dataset_merge, batch_size=batch_size, shuffle=False)

    # Get input size from first batch
    for data in data_loader:
        input_size = data[0].shape[1]
        break

    results = {}

    for model in model_name:
        # print(f"\nTraining {model} model...")

        # Initialize model
        if model == "Encoder_base":
            encoder = Encoder_base(input_size, latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size)
        elif model == 'Encoder_base_2':
            encoder = Encoder_base_2(input_size, latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size)
        elif model == 'Encoder_base_3':
            encoder = Encoder_base_3(input_size, latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size)
        elif model == 'Encoder_base_4':
            encoder = Encoder_base_4(input_size, latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size // 2)
        elif model == 'Encoder_base_5':
            encoder = Encoder_base_5(input_size, latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size // 4)
        elif model == "LSTM":
            encoder = EncoderLSTM(input_size, hidden_dim, latent_space_size, num_layers=1)
            decoder = Decoder_base(input_size, latent_space_size)
        elif model == "GradKPCA":
            encoder = GradKPCA(input_size, n_components, latent_space_size, batch_size)
            decoder = Decoder_base(input_size, latent_space_size)
        elif model == "GradPCA":
            encoder = GradPCA(input_size, n_components, latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size)
        elif model == "GradTensorCP":
            encoder = GradTensorCP(seq_len=window_size,input_size=int(input_size/window_size), n_ranks=4, latent_space_size=latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size)
        elif model == "GradDict":
            encoder = GradDict(input_size, n_components, latent_space_size)
            decoder = Decoder_base(input_size, latent_space_size)
        else:
            raise ValueError(f"Unsupported model_name: {model}")


        encoder.to(device)
        decoder.to(device)

        optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
        mse_loss = nn.MSELoss(reduction='none')
        all_reconstructed_data = []
        epoch_losses = []
        mse_sum = 0

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            encoder.train()
            decoder.train()

            batch_idx = 0
            for data in data_loader:
                data = data[0].float().to(device)

                if model == 'GradTensorCP':
                    data_tensor = reshape_to_3d(data,window_size)
                    latent_space = encoder(data_tensor)
                    reconstruction = decoder(latent_space)
                else:
                    latent_space = encoder(data)
                    reconstruction = decoder(latent_space)

                if model == 'GradKPCA' or model == 'GradPCA' or model == 'GradTensorCP' or model == 'GradDict':
                    loss = encoder.self_loss(reconstruction)
                else:
                    loss = mse_loss(reconstruction, data).mean()
                total_loss += loss.item()
                num_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch == epochs - 1:
                    mse_sum += mse_loss(reconstruction, data).sum().item()
                    all_reconstructed_data.append(reconstruction.detach().cpu().numpy())

                # if batch_idx % 100 == 0:
                #     print(
                #         f'Model {model}, Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.mean().item()}')
                batch_idx += 1

            average_epoch_loss = total_loss / num_batches
            epoch_losses.append(average_epoch_loss)

        reconstructed_data = np.vstack(all_reconstructed_data)

        # Store results for this model
        results[model] = {
            'encoder': encoder,
            'decoder': decoder,
            'reconstructed_data': reconstructed_data,
            'epoch_losses': epoch_losses,
            'mse_sum': mse_sum
        }

    # print('\nTraining completed for all models.')
    return results

def reshape_to_3d(data, seq_len):
    """
    Convert 2D array of shape (n, p) to 3D tensor of shape (n, seq_len, num_features).
    """

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    n, p = data.shape

    if p % seq_len != 0:
        raise ValueError(f"Feature dimension p ({p}) must be divisible by seq_len ({seq_len}).")

    num_features = p // seq_len

    reshaped_data = data.view(n, seq_len, num_features)

    return reshaped_data


def normalize_tensor(data_train_tensor):
    """
    Normalize 3D tensor (B, L, C)
    """
    mean_tensor = data_train_tensor.mean(dim=1, keepdim=True)
    normalized_tensor = data_train_tensor - mean_tensor
    return normalized_tensor

def update_models_with_online_gradient_descent(feature_extractor,reconstruction_extractor,y_pred,data_seg_3d,epochs,learning_rate,batch_size,window_size):
    """
    Update parameters of feature_extractor and reconstruction_extractor.
    Remove data points where y_pred equals 1 from data_seg_3d and update model parameters using online gradient descent.
    """
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.bool)
    anomaly_indices = torch.where(y_pred_tensor == 1)[0]
    mask = torch.ones_like(data_seg_3d, dtype=torch.bool)

    for t in anomaly_indices:
        mask[t, -1, :] = False
        for offset in range(1, window_size):
            if t + offset < len(data_seg_3d):
                mask[t + offset, -(offset + 1), :] = False

    def custom_mse_loss(pred, target, mask):
        loss = (pred - target).pow(2) * mask
        return loss.sum() / mask.sum()

    filtered_data = data_seg_3d * mask
    filtered_data_update = filtered_data.view(data_seg_3d.shape[0],-1)
    mask = mask.view(data_seg_3d.shape[0],-1)
    # print("filtered_data:",filtered_data_update.shape)

    encoder, decoder = reconstruction_extractor
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        filtered_data_update = filtered_data_update.float()
        filtered_feature = encoder(filtered_data_update)
        reconstructed_data = decoder(filtered_feature)
        # print("reconstructed_data:",reconstructed_data.shape)

        loss = custom_mse_loss(reconstructed_data, filtered_data_update, mask)

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    feature_extractor_new = encoder
    reconstruction_extractor_new = (encoder, decoder)
    return feature_extractor_new, reconstruction_extractor_new, filtered_data

def weighted_reconstruction(data_seg_3d, recon_experts, weights):
    """
    Reconstruct input data using weighted models.
    """
    data_seg_3d = data_seg_3d.float()
    data_seg_3d = data_seg_3d.view(data_seg_3d.shape[0],-1)
    # print("data_seg_3d:",data_seg_3d.shape)
    weights = torch.tensor(weights, dtype=torch.float32)

    weighted_recon = torch.zeros_like(data_seg_3d)
    weight_idx = 0
    weight_idx = 0
    for model_experts in recon_experts:  # model_experts is [(Encoder, Decoder), ...]
        for encoder, decoder in model_experts:  # Directly unpack tuple
            latent = encoder(data_seg_3d)
            expert_reconstruction = decoder(latent)
            weighted_recon += weights[weight_idx] * expert_reconstruction
            weight_idx += 1
    return weighted_recon.detach().cpu().numpy()