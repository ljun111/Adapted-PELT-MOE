import torch
import os
import numpy as np
import yaml
from Online_PELT_MOE_separated.Encoder_train import train_encoder_decoder,reshape_to_3d
from Online_PELT_MOE_separated.overall_model import train_and_segment_online
from Plot.plot import plot_data_and_reconstructions, plot_loss, plot_reconstruction_data_with_changepoints, plot_features_with_anomalies, plot_sse
from utils.utils import standardize_data, downsample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir(r'D:\lym本科\科研相关\华为项目资料\2025-5-21\Adapted-PELT-MOE')   # set as your own working directory
configfile = r'.\cfgs\AIOps.yaml'   # change this to detect different dataset

with open(configfile, 'r') as file:
    params = yaml.safe_load(file)

if __name__ == '__main__':
    dataset_name = params.get('dataset')
    down_size = params.get('down_size')
    batch_size = params.get('batch_size')
    shuffle = params.get('shuffle')
    num_workers = params.get('num_workers')
    latent_space_size = params.get('latent_space_size')
    hidden_dim = params.get('hidden_dim')
    epochs = params.get('epochs')
    learning_rate = params.get('learning_rate')
    min_seg_len = params.get('min_seg_len')
    K = params.get('K')
    beta = params.get('beta')
    quantile = params.get('quantile')
    method = params.get('method')
    window_size = params.get('window_size')
    model_name = params.get('model_name')
    gamma = params.get('gamma')
    tau = params.get('tau')
    lambda1 = params.get('lambda1')
    lambda2 = params.get('lambda2')
    lambda3 = params.get('lambda3')
    n_components = params.get('n_components')

    npy_file_path_train = f'./data/{dataset_name}/train_{window_size}_v2.npy'
    # npy_file_path_train = f'./data/{dataset_name}/Fourier_series_200_train_v1.npy'
    data_train = np.load(npy_file_path_train, allow_pickle=True)
    print("data_train:",data_train.shape)
    data_train = data_train.astype(np.float32)
    data_train, mean, std = standardize_data(data_train)


    # npy_file_path_test = f'./data/{dataset_name}/Fourier_series_200_test_v1.npy'
    # npy_file_path_test_label = f'./data/{dataset_name}/Fourier_series_200_test_label_v1.npy'
    npy_file_path_test = f'./data/{dataset_name}/test_{window_size}_v2.npy'
    npy_file_path_test_label = f'./data/{dataset_name}/test_label_{window_size}_v2.npy'
    data_test = np.load(npy_file_path_test, allow_pickle=True)
    data_test = data_test.astype(np.float32)
    print("data_test:", data_test.shape)
    data_test = (data_test - mean) / std
    data_test_label = np.load(npy_file_path_test_label, allow_pickle=True)
    print("Anomaly_rate:",sum(data_test_label)/len(data_test_label))

    # Downsample
    data_train = downsample(data_train, labels=None, down_size=down_size)
    print("Training shape after Downsample:", data_train.shape)
    data_test, data_test_label = downsample(data_test, data_test_label, down_size)
    print("Testing shape after Downsample:", data_test.shape)
    print("Testing label shape after Downsample:", data_test_label.shape)

    # APM_train
    changepoints_online = train_and_segment_online(model_name, data_train, data_test, latent_space_size, hidden_dim,
                                                           window_size, min_seg_len, K, beta, gamma, epochs, learning_rate,
                                                           batch_size, tau, lambda1, lambda2, lambda3, n_components)

    # # one model
    # results = train_encoder_decoder(model_name, data_test[:300], epochs,learning_rate,latent_space_size, hidden_dim,batch_size, window_size, n_components)
    # print("SSE:",results['Encoder_base']['mse_sum'])
    # plot_loss(results['Encoder_base']['epoch_losses'])
    # plot_reconstruction_data_with_changepoints(data_train[:, :8],
    #                                            results['Encoder_base']['reconstructed_data'][:, :9], changepoints=[],
    #                                            changepoints_type=[])
    # plot_reconstruction_data_with_changepoints(data_test[:,:8],
    #                                            results['Encoder_base']['reconstructed_data'][:, :8], changepoints=[],
    #                                            changepoints_type=[])

    # print("reconstructed_data:", reconstructed_data.shape)
    # # print("encoder parameters:", list(encoder.named_parameters()))
    # print("decoder parameters:", list(decoder.named_parameters()))
    # reconstructed_data_test = []
    # recon_loss = []
    # with torch.no_grad():
    #     encoder.eval()
    #     decoder.eval()
    #     data_test_tensor = reshape_to_3d(data_test, window_size)
    #     # data_test_tensor = normalize_tensor(data_test_tensor)
    #     data_test_tensor = data_test_tensor.view(data_test.shape[0], -1)
    #     data_test_tensor = data_test_tensor.to(device)
    #     feature_outputs = encoder(data_test_tensor)
    #     reconstructed_data_output = decoder(feature_outputs)
    #     reconstructed_data_test.append(reconstructed_data_output.detach().cpu().numpy())
    #
    #     data_test_tensor_new = reshape_to_3d(data_test, window_size)
    #     # data_test_tensor_new = normalize_tensor(data_test_tensor_new)
    #     first_time_step_data = data_test_tensor_new[:, -1:, :]
    #     first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
    #     first_time_step_data = first_time_step_data.view(data_test_tensor_new.shape[0], -1)
    #     first_time_step_data = first_time_step_data.to(device)
    #     latent_space = encoder(first_time_step_data)
    #     for i, layer in enumerate(decoder.children()):
    #         latent_space = layer(latent_space)
    #         print(f"Output of layer {i}:")
    #         print(latent_space)
    #     latent_space_1 = encoder(first_time_step_data)
    #     reconstruction = decoder(latent_space_1)
    #     # print("reconstruction:",reconstruction)
    #     # print("Is input contiguous?", first_time_step_data.is_contiguous())
    #     # print("encoder mode:", encoder.training)
    #     # print("decoder mode:", decoder.training)
    #     # print(next(encoder.parameters()).device)
    #     # print(next(decoder.parameters()).device)
    #     # print(first_time_step_data.device)
    #     # print(first_time_step_data.dtype)
    #
    #     reconstruction_error = torch.nn.functional.mse_loss(
    #         reconstruction,
    #         first_time_step_data,
    #         reduction='none'
    #     ).mean(dim=1) / window_size
    #     recon_loss.extend(reconstruction_error.detach().cpu().numpy())
    # reconstructed_data_test_array = np.array(reconstructed_data_test)
    # reconstructed_data_test_array = np.squeeze(reconstructed_data_test_array)
    # print("reconstructed_data:", reconstructed_data_test_array.shape)
    # print("origin_data:",first_time_step_data[:20,])
    # print("reconstructed_data:",reconstruction[:20,])
    # print("latent_space:",latent_space)
    # recon_loss = np.array(recon_loss)
    # print("recon_loss:", recon_loss.shape)
    # print("recon_loss:",recon_loss[:20])
    # data_test_tensor = reshape_to_3d(data_test, window_size)
    # # data_test_tensor = normalize_tensor(data_test_tensor)
    # data_test_tensor_array = np.array(data_test_tensor)
    # data_test_tensor_array = data_test_tensor_array.reshape(data_test_tensor_array.shape[0], data_train.shape[1])
    #
    # data_train_tensor = reshape_to_3d(data_train, window_size)
    # # data_train_tensor = normalize_tensor(data_train_tensor)
    # data_train_tensor_array = np.array(data_train_tensor)
    # data_train_tensor_array = data_train_tensor_array.reshape(data_train_tensor_array.shape[0], data_train.shape[1])
    #
    # plot_reconstruction_data_with_changepoints(data_train_tensor_array[:, :9],
    #                                            reconstructed_data[:, :9], changepoints=[],
    #                                            changepoints_type=[])
    # plot_reconstruction_data_with_changepoints(data_test_tensor_array[:, :13], reconstructed_data_test_array[:, :13],
    #                                            changepoints=[], changepoints_type=[])
    # plot_loss(epoch_losses)
    # plot_features_with_anomalies(data_test[:, :8], data_test_label)
    # plot_sse(recon_loss, changepoints=[], labels=data_test_label)
