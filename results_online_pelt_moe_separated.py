import torch
import os
import numpy as np
import yaml
from Plot.plot import plot_candidates, plot_predictions_with_mismatches, plot_data_with_changepoints, plot_data_and_reconstructions, plot_reconstruction_data_with_changepoints, plot_data_with_truechangepoints
from Online_PELT_MOE_separated.Encoder_train import reshape_to_3d, normalize_tensor
from Online_PELT_MOE_separated.online_detection_separated import process_models_reconstruction
from utils.utils import standardize_data, downsample
from metric.metric import precision, recall, f1_score, calculate_accuracy, calculate_auroc, calculate_auprc, adjusted_f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir(r'D:\lym本科\科研相关\华为项目资料\2025-5-21\Adapted-PELT-MOE')   # set as your own working directory
configfile = r'.\cfgs\AIOps.yaml'   # change this to detect different dataset

with open(configfile, 'r') as file:
    params = yaml.safe_load(file)

if __name__ == '__main__':
    dataset_name = params.get('dataset')
    latent_space_size = params.get('latent_space_size')
    down_size = params.get('down_size')
    window_size = params.get('window_size')
    tau = params.get('tau')
    quantile = params.get('quantile')
    model_name = params.get('model_name')

    origin_data_path = f'./data/{dataset_name}/train_{window_size}_v2.npy'
    # origin_data_path = f'./data/{dataset_name}/Fourier_series_200_train_v1.npy'
    origin_data = np.load(origin_data_path, allow_pickle=True)
    origin_data = origin_data.astype(np.float32)
    origin_data, mean, std = standardize_data(origin_data)

    # npy_file_path_test = f'./data/{dataset_name}/Fourier_series_200_test_v1.npy'
    # npy_file_path_test_label = f'./data/{dataset_name}/Fourier_series_200_test_label_v1.npy'
    npy_file_path_test = f'./data/{dataset_name}/test_{window_size}_v2.npy'
    npy_file_path_test_label = f'./data/{dataset_name}/test_label_{window_size}_v2.npy'
    data_test = np.load(npy_file_path_test, allow_pickle=True)
    data_test = data_test.astype(np.float32)
    data_test = (data_test - mean) / std
    data_test_label = np.load(npy_file_path_test_label, allow_pickle=True)

    # Downsample
    origin_data = downsample(origin_data, labels=None, down_size=down_size)
    print("Training shape after Downsample:", origin_data.shape)
    data_test, data_test_label = downsample(data_test, data_test_label, down_size)
    print("Testing shape after Downsample:", data_test.shape)
    print("Testing label shape after Downsample:", data_test_label.shape)

    # # Fourier data
    # data_test_label_fourier = np.zeros_like(data_test_label)
    # for i in range(len(data_test_label)):
    #     if data_test_label[i] == 1:
    #         if i + (window_size - 1) < len(data_test_label_fourier):
    #             data_test_label_fourier[i-window_size+1] = 1
    # data_path = f'.\models\segmentation_results_{data_test.shape[0]}_{model_name}.npz'
    # data = np.load(data_path, allow_pickle=True)
    # changepoints = data['changepoints']
    # candidate_score = data['candidate_score']
    # changepoint_type = data['changepoint_type']
    # plot_candidates(changepoints=[], candidate_score=candidate_score, candidate_recon_loss=[], candidate_likelihood_value=[],
    #                 candidate_cos_sim=[], changepoint_type=[], data_test_label=data_test_label_fourier)
    # plot_data_with_truechangepoints(data_test[:,0:1], changepoints=[0], true_changepoints=[41,81,123,165])
    # plot_data_with_truechangepoints(data_test[:, 0:1], changepoints=changepoints, true_changepoints=[41, 81, 123, 165])
    # y_pred_true = []
    # anomaly_labels_true = []
    # true_F1_score = 0
    # threshold = np.quantile(candidate_score,q=quantile/100.0)
    # print("threshold:",threshold)
    # for candidate in candidate_score:
    #     if candidate < threshold:
    #         anomaly_labels_true.append(0)
    #     else:
    #         anomaly_labels_true.append(1)
    # print("data_test_label_fourier:",data_test_label_fourier)
    # print("anomaly_labels_true:",anomaly_labels_true)
    # f1_score_true = f1_score(data_test_label_fourier, anomaly_labels_true)
    # precision_true = precision(data_test_label_fourier, anomaly_labels_true)
    # recall_true = recall(data_test_label_fourier, anomaly_labels_true)
    # print("True_F1_Score:",f1_score_true)
    # print("True_Precision:", precision_true)
    # print("True_Recall:", recall_true)
    # print("True_AUROC:",calculate_auroc(data_test_label_fourier,candidate_score))
    # print("True_AUPRC:", calculate_auprc(data_test_label_fourier, candidate_score))
    # print("PA_ACC_PRE_REC_F1:",adjusted_f1_score(data_test_label_fourier,anomaly_labels_true))


    # public dataset
    candidate_score = process_models_reconstruction(model_name, data_test, top_k=3)
    plot_candidates(changepoints=[], candidate_score=candidate_score, candidate_recon_loss=[], candidate_likelihood_value=[],
                    candidate_cos_sim=[], changepoint_type=[], data_test_label=data_test_label)

    y_pred_true = []
    anomaly_labels_true = []
    true_F1_score = 0
    threshold = np.quantile(candidate_score,q=quantile/100.0)
    print("threshold:",threshold)
    for candidate in candidate_score:
        if candidate < threshold:
            anomaly_labels_true.append(0)
        else:
            anomaly_labels_true.append(1)

    f1_score_true = f1_score(data_test_label, anomaly_labels_true)
    precision_true = precision(data_test_label, anomaly_labels_true)
    recall_true = recall(data_test_label, anomaly_labels_true)
    print("True_F1_Score:",f1_score_true)
    print("True_Precision:", precision_true)
    print("True_Recall:", recall_true)
    print("True_AUROC:",calculate_auroc(data_test_label,candidate_score))
    print("True_AUPRC:", calculate_auprc(data_test_label, candidate_score))
    print("PA_ACC_PRE_REC_F1:",adjusted_f1_score(data_test_label,anomaly_labels_true))
